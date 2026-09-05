"""best 更新時の検証プレビュー画像(3x3)を描画する共有モジュール。

val から 40° 刻みの目標 yaw に最も近い 9 行を決定的に選び(毎回同じ顔)、
各セルに顔画像と円環インジケーターを描く:
  - 円環は俯瞰ビュー(カメラは下)。針は 下 = 0°(正面)、左 = +90°
    (画面左向き)、上 = 180°(背面)、右 = 270°
  - GT = 緑、予測 = 赤。左上に数値(gt / pred / err)
出力は 1 枚(runs/<run>/val_preview_best.png、best 更新のたびに上書き)。
"""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import torch

CELL = 224
GRID = 3


def _circ_diff(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def select_indices(ds, n: int = GRID * GRID) -> list[int]:
    """目標 yaw(360/n 刻み)に最も近い val 行を重複なしで選ぶ(決定的)。"""
    yaws = [float(r["yaw_deg"]) for r in ds.rows]
    chosen: list[int] = []
    for k in range(n):
        target = k * 360.0 / n
        order = sorted(range(len(yaws)), key=lambda i: _circ_diff(yaws[i], target))
        idx = next(i for i in order if i not in chosen)
        chosen.append(idx)
    return chosen


def _needle(img: np.ndarray, center: tuple[int, int], radius: int,
            yaw_deg: float, color: tuple[int, int, int], thickness: int) -> None:
    rad = math.radians(yaw_deg)
    # 俯瞰ビュー: 画面ベクトル = (-sin(yaw), +cos(yaw))(x 右, y 下)
    dx, dy = -math.sin(rad), math.cos(rad)
    tip = (int(center[0] + dx * radius), int(center[1] + dy * radius))
    cv2.arrowedLine(img, center, tip, color, thickness, cv2.LINE_AA, tipLength=0.25)


def _draw_cell(im_bgr: np.ndarray, gt: float, pred: float) -> np.ndarray:
    cell = cv2.resize(im_bgr, (CELL, CELL), interpolation=cv2.INTER_LINEAR)
    # 円環(右下、半透明の下地)
    radius = 36
    center = (CELL - radius - 8, CELL - radius - 8)
    overlay = cell.copy()
    cv2.circle(overlay, center, radius + 6, (255, 255, 255), -1, cv2.LINE_AA)
    cell = cv2.addWeighted(overlay, 0.55, cell, 0.45, 0)
    cv2.circle(cell, center, radius, (60, 60, 60), 2, cv2.LINE_AA)
    # 0°(正面 = カメラ側)の目盛りを下側に打つ
    cv2.circle(cell, (center[0], center[1] + radius), 3, (60, 60, 60), -1, cv2.LINE_AA)
    _needle(cell, center, radius - 4, gt, (0, 200, 0), 3)      # GT: 緑
    _needle(cell, center, radius - 4, pred, (0, 0, 255), 2)    # 予測: 赤
    err = _circ_diff(gt, pred)
    for i, (text, color) in enumerate([
            (f"gt   {gt:6.1f}", (0, 200, 0)),
            (f"pred {pred:6.1f}", (0, 0, 255)),
            (f"err  {err:6.1f}", (255, 255, 255))]):
        cv2.putText(cell, text, (6, 18 + 18 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(cell, text, (6, 18 + 18 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return cell


@torch.no_grad()
def render_val_preview(model: torch.nn.Module, ds, indices: list[int],
                       out_path: str | Path, device: str, epoch: int,
                       maae: float,
                       amp_dtype: torch.dtype = torch.float16) -> None:
    """9 サンプルを推論して 3x3 プレビューを書き出す(model は eval 済み前提)。"""
    xs = torch.stack([ds[i][0] for i in indices]).to(device)
    with torch.autocast("cuda", dtype=amp_dtype):
        pred = model(xs).float().cpu()
    pred_deg = (torch.rad2deg(torch.atan2(pred[:, 1], pred[:, 0])) % 360.0).tolist()

    cells: list[np.ndarray] = []
    for i, p in zip(indices, pred_deg):
        r = ds.rows[i]
        im = cv2.imread(str(ds.root / r["image"]), cv2.IMREAD_COLOR)
        cells.append(_draw_cell(im, float(r["yaw_deg"]), float(p)))
    rows = [np.hstack(cells[k * GRID:(k + 1) * GRID]) for k in range(GRID)]
    grid = np.vstack(rows)

    header = np.full((28, grid.shape[1], 3), 32, np.uint8)
    cv2.putText(header, f"epoch {epoch}  val maae {maae:.2f} deg  "
                        f"(GT=green, pred=red; ring: down=0 front, left=+90)",
                (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_path), np.vstack([header, grid]))
