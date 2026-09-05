"""Yaw 回帰用 augmentation コア(HRFFA D4 幾何拡張の yaw-only 適応版)。

HRFFA (High-Angle_Robust_Fast_FaceAlignment) の設計を踏襲:
すべての幾何拡張を 1 つの 3x3 射影変換 T に合成し、画像は 1 回だけ warp する
(リサイズも warp に含まれるため、学習前処理の補間は INTER_LINEAR に統一される)。

yaw のみをラベルとして持つため、GT 更新は次のとおり
(規約: 画像 x 右, y 下, カメラ z 奥。yaw は sixdrepnet360 規約、roll ≈ 0 の合成データ前提):
  - **水平反転**: R' = M R M (M = diag(-1,1,1)) は yaw / roll の符号反転に相当
      → yaw' = -yaw(厳密)。
  - **カメラ方位回転**(ψ): H = K Ry(ψ) K^{-1} の純カメラ回転ワープ。
      HRFFA は完全な回転行列 GT を R' = Ry(ψ) R と厳密更新するが、本タスクの yaw は
      6DRepNet 系の appearance 基準規約であり、クロップ済み頭部への純カメラ回転は
      頭部の見えを変えない(新たに見える面が無い)ため yaw はほぼ不変。
      実測(scripts/verify_cam_yaw_sign.py、sixdrepnet360、ψ=±15°、focal_ratio 1.2)では
      読み値が +0.166·ψ だけ動くため、yaw' = yaw + CAM_YAW_COEF·ψ と補正する。
  - **カメラ俯仰回転**(φ): R' = Rx(φ) R。yaw の euler 抽出値は |yaw| が大きい領域で
      わずかに変わるが、|φ| を小さく保ち yaw 不変と近似する。
  - **小角 roll**(画像内回転 θ): R' = Rz(θ) R。full-360 roll は yaw の euler 抽出値を
      大きく変えるため使わない。|θ| 小のとき yaw 不変と近似する。
  - スケール・平行移動は yaw を変えない。

photometric / motion blur / random erase は HRFFA data/dataset.py の実装を移植
(いずれも yaw GT 不変)。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

# Ry(+ψ) ワープが appearance 基準 yaw に与える実測係数
# (scripts/verify_cam_yaw_sign.py: ψ=±15° で mean_delta ≈ ±2.5° → 2.49/15)
CAM_YAW_COEF: float = 0.166


def _rx(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def _ry(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


@dataclass
class GeometricParams:
    """1 サンプル分の幾何拡張パラメータ(決定済みの値)。"""
    out_size: int = 128
    roll_deg: float = 0.0        # 小角の画像内回転(yaw 不変近似)
    cam_pitch_deg: float = 0.0   # カメラ俯仰(yaw 不変近似)
    cam_yaw_deg: float = 0.0     # カメラ方位(yaw' = yaw + CAM_YAW_COEF * ψ)
    scale: float = 1.0
    tx: float = 0.0              # 出力サイズ比の平行移動
    ty: float = 0.0
    hflip: bool = False
    focal_ratio: float = 1.2


@dataclass
class GeometricPolicy:
    """サンプリング範囲(学習設定)。入力は既に 5% マージン付き正方クロップ。"""
    out_size: int = 128
    roll_deg: float = 8.0
    roll_prob: float = 0.3
    cam_pitch_deg: float = 12.0
    cam_yaw_deg: float = 10.0
    cam_prob: float = 0.3
    scale_range: tuple[float, float] = (0.85, 1.1)
    translate: float = 0.05
    hflip_prob: float = 0.5
    focal_ratio: float = 1.2

    def sample(self, rng: np.random.Generator) -> GeometricParams:
        use_roll = rng.random() < self.roll_prob
        use_cam = rng.random() < self.cam_prob
        return GeometricParams(
            out_size=self.out_size,
            roll_deg=float(rng.uniform(-self.roll_deg, self.roll_deg)) if use_roll else 0.0,
            cam_pitch_deg=(float(rng.uniform(-self.cam_pitch_deg, self.cam_pitch_deg))
                           if use_cam else 0.0),
            cam_yaw_deg=(float(rng.uniform(-self.cam_yaw_deg, self.cam_yaw_deg))
                         if use_cam else 0.0),
            scale=float(rng.uniform(*self.scale_range)),
            tx=float(rng.uniform(-self.translate, self.translate)),
            ty=float(rng.uniform(-self.translate, self.translate)),
            hflip=bool(rng.random() < self.hflip_prob),
            focal_ratio=self.focal_ratio,
        )


def crop_affine(w: int, h: int, p: GeometricParams) -> np.ndarray:
    """入力画像全体(= 5% マージン済み head crop)を out_size 正方へ写す相似変換(3x3)。

    回転・スケール・平行移動もクロップ中心基準でここに合成する。
    """
    cx, cy = w / 2, h / 2
    side = max(w, h)
    s = p.out_size / side * p.scale
    theta = math.radians(p.roll_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    half = p.out_size / 2
    T = np.array([
        [s * cos_t, -s * sin_t, 0.0],
        [s * sin_t, s * cos_t, 0.0],
        [0.0, 0.0, 1.0],
    ])
    T[0, 2] = half + p.tx * p.out_size - (T[0, 0] * cx + T[0, 1] * cy)
    T[1, 2] = half + p.ty * p.out_size - (T[1, 0] * cx + T[1, 1] * cy)
    return T


def camera_homography(p: GeometricParams) -> np.ndarray:
    """純カメラ回転のホモグラフィ H(出力クロップ座標系)。"""
    phi = math.radians(p.cam_pitch_deg)
    psi = math.radians(p.cam_yaw_deg)
    R_cam = _ry(psi) @ _rx(phi)
    f = p.focal_ratio * p.out_size
    c = p.out_size / 2
    K = np.array([[f, 0, c], [0, f, c], [0, 0, 1.0]])
    return K @ R_cam @ np.linalg.inv(K)


def flip_matrix(out_size: int) -> np.ndarray:
    return np.array([[-1.0, 0.0, out_size - 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])


def apply_geometric(image: np.ndarray, yaw_deg: float,
                    p: GeometricParams) -> tuple[np.ndarray, float]:
    """幾何拡張を適用し、(出力画像, 更新後 yaw_deg) を返す。画像の warp は 1 回だけ。"""
    h, w = image.shape[:2]
    T = crop_affine(w, h, p)
    yaw = yaw_deg

    if abs(p.cam_pitch_deg) > 1e-9 or abs(p.cam_yaw_deg) > 1e-9:
        H = camera_homography(p)
        T = H @ T
        # カメラ回転は f·tanψ 級の平行移動成分を持つため、クロップ中心を出力中心へ
        # 戻す(クロップ窓の平行移動であり yaw GT には影響しない)
        m = T @ np.array([w / 2, h / 2, 1.0])
        mx, my = m[0] / m[2], m[1] / m[2]
        half = p.out_size / 2
        recenter = np.array([[1.0, 0.0, half + p.tx * p.out_size - mx],
                             [0.0, 1.0, half + p.ty * p.out_size - my],
                             [0.0, 0.0, 1.0]])
        T = recenter @ T
        yaw = yaw + CAM_YAW_COEF * p.cam_yaw_deg

    if p.hflip:
        T = flip_matrix(p.out_size) @ T
        yaw = -yaw

    out = cv2.warpPerspective(
        image, T, (p.out_size, p.out_size),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return out, yaw % 360.0


# ---------------------------------------------------------------------------
# photometric 系(HRFFA data/dataset.py より移植。yaw GT 不変)
# ---------------------------------------------------------------------------

def photometric(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """軽量 photometric 拡張(RGB uint8 → RGB uint8)。"""
    x = img.astype(np.float32)
    if rng.random() < 0.8:  # 明度・コントラスト・ガンマ
        x = x * rng.uniform(0.6, 1.4) + rng.uniform(-25, 25)
        x = 255.0 * (x.clip(0, 255) / 255.0) ** rng.uniform(0.7, 1.4)
    if rng.random() < 0.2:  # グレースケール化
        g = cv2.cvtColor(x.clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        x = cv2.cvtColor(g, cv2.COLOR_GRAY2RGB).astype(np.float32)
    if rng.random() < 0.3:  # ガウスノイズ
        x = x + rng.normal(0, rng.uniform(3, 12), x.shape)
    if rng.random() < 0.3:  # ぼかし
        k = int(rng.choice([3, 5]))
        x = cv2.GaussianBlur(x, (k, k), 0)
    x = x.clip(0, 255).astype(np.uint8)
    if rng.random() < 0.3:  # JPEG 劣化
        q = int(rng.integers(35, 85))
        _, enc = cv2.imencode(".jpg", x, [cv2.IMWRITE_JPEG_QUALITY, q])
        x = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return x


def motion_blur(img: np.ndarray, rng: np.random.Generator,
                max_frac: float = 0.06) -> np.ndarray:
    """線形モーションブラー(長さ = 辺の 1〜max_frac、角度一様)。GT は不変。"""
    h, w = img.shape[:2]
    length = int(max(3, rng.uniform(0.01, max_frac) * max(h, w)))
    length += (length + 1) % 2
    k = np.zeros((length, length), np.float32)
    c = length // 2
    th = rng.uniform(0, np.pi)
    for t in np.linspace(-c, c, length * 4):
        x, y = int(round(c + t * np.cos(th))), int(round(c + t * np.sin(th)))
        if 0 <= x < length and 0 <= y < length:
            k[y, x] = 1.0
    k /= max(k.sum(), 1.0)
    return cv2.filter2D(img, -1, k)


def random_erase(img: np.ndarray, rng: np.random.Generator,
                 n_max: int = 2) -> np.ndarray:
    """矩形領域をノイズ/平均色で消去する遮蔽拡張(yaw GT は不変。
    合成遮蔽下でも向きを当てる学習を意図)。"""
    h, w = img.shape[:2]
    out = img.copy()
    for _ in range(int(rng.integers(1, n_max + 1))):
        ew = int(w * rng.uniform(0.10, 0.28))
        eh = int(h * rng.uniform(0.10, 0.28))
        x0 = int(rng.integers(0, max(w - ew, 1)))
        y0 = int(rng.integers(0, max(h - eh, 1)))
        if rng.random() < 0.5:
            out[y0:y0 + eh, x0:x0 + ew] = rng.integers(
                0, 256, (eh, ew, 3), dtype=np.uint8)
        else:
            out[y0:y0 + eh, x0:x0 + ew] = out[y0:y0 + eh, x0:x0 + ew].mean(
                axis=(0, 1), keepdims=True).astype(np.uint8)
    return out


def lowres_jitter(img: np.ndarray, rng: np.random.Generator,
                  min_side: int = 20) -> np.ndarray:
    """縮小→再拡大で低解像度ドメイン(監視カメラ)を模擬。GT は不変。
    補間は学習前処理の方針に合わせ INTER_LINEAR に統一。"""
    h = img.shape[0]
    lo = int(rng.integers(min_side, max(min_side + 1, h)))
    small = cv2.resize(img, (lo, lo), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (h, h), interpolation=cv2.INTER_LINEAR)
