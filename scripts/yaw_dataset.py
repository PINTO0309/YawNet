"""Dataset / sampler for yawpose (360-degree yaw, biternion targets).

augmentation は scripts/augment.py(HRFFA D4 の yaw-only 適応版)に集約:
  - 幾何: 1 つの 3x3 行列に合成して 1 回だけ warp
    (小角 roll / カメラ回転ホモグラフィ / flip / スケール / 平行移動。
     リサイズも warp に含まれ INTER_LINEAR に統一)
  - photometric / motion blur / random erase / low-res jitter(いずれも HRFFA 移植)
評価時は決定的な INTER_LINEAR リサイズのみ。
"""
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from augment import (GeometricPolicy, apply_geometric, lowres_jitter,
                     motion_blur, photometric, random_erase)

# 既定は center05 正規化: ((x/255) - 0.5) / 0.5 → 値域 [-1, 1]。
# DINOv3 教師(train_teacher_dinov3.py / 蒸留の教師ビュー)のみ ImageNet 正規化を使う
# (DINOv3 hub backbone の要件)。sixdrepnet / hrffa の検証スクリプトも同様に ImageNet。
NORM_MEAN = 0.5
NORM_STD = 0.5
NORMS: dict[str, tuple[np.ndarray, np.ndarray]] = {
    "center05": (np.array([0.5, 0.5, 0.5], np.float32),
                 np.array([0.5, 0.5, 0.5], np.float32)),
    "imagenet": (np.array([0.485, 0.456, 0.406], np.float32),
                 np.array([0.229, 0.224, 0.225], np.float32)),
}


def normalize_image(im: np.ndarray, norm: str) -> torch.Tensor:
    mean, std = NORMS[norm]
    arr = (im.astype(np.float32) / 255.0 - mean) / std
    return torch.from_numpy(arr.transpose(2, 0, 1).copy())


class YawDataset(Dataset):
    def __init__(self, root: str | Path, split: str, size: int, train: bool = False,
                 policy: GeometricPolicy | None = None,
                 motion_blur_prob: float = 0.2,
                 erase_prob: float = 0.25,
                 lowres_prob: float = 0.35,
                 input_norm: str = "center05",
                 seed: int = 0) -> None:
        self.root = Path(root)
        self.size = size
        self.train = train
        self.input_norm = input_norm
        self.policy = policy or GeometricPolicy(out_size=size)
        self.policy.out_size = size
        self.motion_blur_prob = motion_blur_prob
        self.erase_prob = erase_prob
        self.lowres_prob = lowres_prob
        self.seed = seed
        # split="unified" は train + val の統合(意図的なデータリーク許容。
        # 最終製品学習など、val をホールドアウトしない用途専用)
        files = ["train.jsonl", "val.jsonl"] if split == "unified" else [f"{split}.jsonl"]
        self.rows: list[dict[str, Any]] = []
        for name in files:
            with open(self.root / name) as f:
                self.rows += [json.loads(line) for line in f if line.strip()]

    def __len__(self) -> int:
        return len(self.rows)

    def bin_weights(self, bin_deg: int = 10, mode: str = "inv",
                    max_ratio: float = 20.0) -> torch.Tensor:
        """yaw ビン頻度に基づくサンプル重み(バランスサンプリング用)。

        mode:
          - "inv" : 1/count       — ビンを完全に平坦化(既定。synthetic_004 補強後の
                                    全周均一分布を前提とした設定。希少ビンは
                                    max_ratio クリップで反復を抑制)
          - "sqrt": 1/sqrt(count) — 部分平坦化
          - "none": 全サンプル等重み(自然分布のまま)
        max_ratio: 重みの最大/最小比の上限。極端に希少なビン(例: 真後ろ帯)への
        過剰集中による同一画像の反復を防ぐためクリップする。
        """
        if mode == "none":
            return torch.ones(len(self.rows), dtype=torch.double)
        if mode not in ("inv", "sqrt"):
            raise ValueError(f"unknown balance mode: {mode}")
        bins = [int(r["yaw_deg"] % 360.0 // bin_deg) for r in self.rows]
        counts = Counter(bins)
        exponent = 1.0 if mode == "inv" else 0.5
        w = np.array([1.0 / (counts[b] ** exponent) for b in bins])
        w = np.minimum(w, w.min() * max_ratio)
        return torch.from_numpy(w)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r = self.rows[idx]
        bgr = cv2.imread(str(self.root / r["image"]), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(self.root / r["image"])
        im = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        yaw = float(r["yaw_deg"])

        if self.train:
            rng = np.random.default_rng(None)
            im, yaw = apply_geometric(im, yaw, self.policy.sample(rng))
            if rng.random() < self.lowres_prob:
                im = lowres_jitter(im, rng)
            im = photometric(im, rng)
            if rng.random() < self.motion_blur_prob:
                im = motion_blur(im, rng)
            if rng.random() < self.erase_prob:
                im = random_erase(im, rng)
        else:
            im = cv2.resize(im, (self.size, self.size), interpolation=cv2.INTER_LINEAR)

        x = normalize_image(im, self.input_norm)
        rad = math.radians(yaw)
        target = torch.tensor([math.cos(rad), math.sin(rad)], dtype=torch.float32)
        return x, target, torch.tensor(yaw, dtype=torch.float32)


class DistillYawDataset(YawDataset):
    """蒸留用の同条件ペア Dataset。

    1 サンプルにつき、幾何変換 + 劣化(photometric / lowres_jitter /
    motion_blur / random_erase)をすべて teacher_size(既定 320)で 1 回だけ
    適用し、同一内容の 2 ビューを返す:
      - 教師ビュー: 劣化適用後の teacher_size 画像
      - 学生ビュー: 同じ画像を size へ縮小しただけのもの
    教師と学生は解像度以外は完全に同条件(同じ幾何変換・同じ劣化)であり、
    yaw ラベル変換(flip の符号反転・カメラ回転の補正)も厳密に一致する。

    返り値: (teacher_x, student_x, target_cos_sin, yaw_deg)
    """

    def __init__(self, root: str | Path, split: str, size: int,
                 teacher_size: int = 320, train: bool = False,
                 policy: GeometricPolicy | None = None,
                 motion_blur_prob: float = 0.2,
                 erase_prob: float = 0.25,
                 lowres_prob: float = 0.35,
                 teacher_norm: str = "center05",
                 input_norm: str = "center05",
                 seed: int = 0) -> None:
        super().__init__(root, split, size, train=train, policy=policy,
                         motion_blur_prob=motion_blur_prob,
                         erase_prob=erase_prob, lowres_prob=lowres_prob,
                         input_norm=input_norm, seed=seed)
        self.teacher_size = teacher_size
        self.teacher_norm = teacher_norm      # DINOv3 教師は "imagenet" を指定する
        self.policy.out_size = teacher_size   # 幾何変換は教師解像度で 1 回だけ行う

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor,
                                             torch.Tensor, torch.Tensor]:
        r = self.rows[idx]
        bgr = cv2.imread(str(self.root / r["image"]), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(self.root / r["image"])
        im = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        yaw = float(r["yaw_deg"])

        if self.train:
            rng = np.random.default_rng(None)
            t_img, yaw = apply_geometric(im, yaw, self.policy.sample(rng))
            # 劣化は教師解像度で 1 回だけ適用し、学生はその縮小版を見る
            # (教師と学生は解像度以外は完全に同条件)
            if rng.random() < self.lowres_prob:
                t_img = lowres_jitter(t_img, rng)
            t_img = photometric(t_img, rng)
            if rng.random() < self.motion_blur_prob:
                t_img = motion_blur(t_img, rng)
            if rng.random() < self.erase_prob:
                t_img = random_erase(t_img, rng)
            s_img = cv2.resize(t_img, (self.size, self.size),
                               interpolation=cv2.INTER_LINEAR)
        else:
            t_img = cv2.resize(im, (self.teacher_size, self.teacher_size),
                               interpolation=cv2.INTER_LINEAR)
            s_img = cv2.resize(im, (self.size, self.size),
                               interpolation=cv2.INTER_LINEAR)

        rad = math.radians(yaw)
        target = torch.tensor([math.cos(rad), math.sin(rad)], dtype=torch.float32)
        return (normalize_image(t_img, self.teacher_norm),
                normalize_image(s_img, self.input_norm),
                target, torch.tensor(yaw, dtype=torch.float32))


def make_balanced_sampler(ds: YawDataset, bin_deg: int = 10, mode: str = "inv",
                          max_ratio: float = 20.0) -> WeightedRandomSampler:
    """yaw ビン頻度バランスの WeightedRandomSampler を作る。

    乱数は torch のグローバル generator を使う(専用 generator は渡さない)。
    train_yawnet.py のチェックポイントが torch のグローバル RNG 状態を保存/復元
    するため、これにより --resume 後もサンプリング列の決定性が保たれる。
    """
    w = ds.bin_weights(bin_deg=bin_deg, mode=mode, max_ratio=max_ratio)
    return WeightedRandomSampler(w, num_samples=len(ds), replacement=True)


def angular_error_deg(pred_unit: torch.Tensor, yaw_deg: torch.Tensor) -> torch.Tensor:
    """Absolute angular error in degrees between predicted unit vector and gt yaw."""
    pred = torch.rad2deg(torch.atan2(pred_unit[:, 1], pred_unit[:, 0])) % 360.0
    diff = (pred - yaw_deg) % 360.0
    return torch.minimum(diff, 360.0 - diff)
