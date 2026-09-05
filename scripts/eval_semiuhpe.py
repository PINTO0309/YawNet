#!/usr/bin/env python3
"""SOTA 比較: SemiUHPE(arXiv 2404.02544)を yawpose val で全周 yaw 評価する。

対象: docs/SemiUHPE の公開重み DAD-WildHead-EffNetV2-S-best.pth
(DAD-3DHeads 学習の全周モデル、matrix Fisher 出力)。

再現する推論経路(predict.py 準拠):
  頭部クロップ(5% マージン想定 = 我々のクロップ規約と一致)
    → PIL resize 224(BICUBIC)→ ImageNet 正規化
    → EffNetV2-S + 独自 classifier(9 出力の Fisher パラメーター A)
    → SVD で回転行列 R へ(batch_torch_A_to_R と同一計算をインライン化)
    → R.T の xyz Euler から yaw = angle[1](DAD 全周ブランチと同一)

yaw 規約の較正: val 内の s001 operator 検証済み行で符号 s∈{+1,-1} を実測し、
ours ≈ s * theirs を確定してから全 val の MAAE / per_bin を算出する。

実行(torchvision / scipy は一時オーバーレイで導入し lock は汚さない):
  uv run --no-sync --with torchvision==0.26.0 --with scipy \
      python scripts/eval_semiuhpe.py
"""
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation
from torch import nn
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
SEMI = ROOT / "docs" / "SemiUHPE"
CKPT = SEMI / "weights" / "DAD-WildHead-EffNetV2-S-best.pth"
VAL = ROOT / "data" / "yawpose"
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def signed(deg: float) -> float:
    d = deg % 360.0
    return d - 360.0 if d > 180.0 else d


def limit_angle(angle: float, pi: float = 180.0) -> float:
    """SemiUHPE src/utils.py の limit_angle と同一。"""
    if angle < -pi:
        k = -2 * (int(angle / pi) // 2)
        angle = angle + k * pi
    if angle > pi:
        k = 2 * ((int(angle / pi) + 1) // 2)
        angle = angle - k * pi
    return angle


def build_model() -> nn.Module:
    """networks.py get_EfficientNet_V2(config, "S") と同一構造(num_classes=9)。"""
    from torchvision import models  # noqa: PLC0415
    model = models.efficientnet_v2_s(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(1280, 512),
        nn.BatchNorm1d(512),
        nn.ReLU6(inplace=True),
        nn.Linear(512, 128),
        nn.BatchNorm1d(128),
        nn.ReLU6(inplace=True),
        nn.Linear(128, 9),
    )
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model_state_dict"])
    return model.cuda().eval()


def batch_A_to_R(A: torch.Tensor) -> torch.Tensor:
    """fisher_utils.batch_torch_A_to_R と同一計算(依存を持ち込まずインライン化)。"""
    A = A.reshape(-1, 3, 3).cpu()
    U, S, V = torch.svd(A)
    with torch.no_grad():
        s3sign = torch.det(torch.matmul(U, V.transpose(1, 2)))
    U = torch.cat((U[:, :, :2], U[:, :, 2:] * s3sign[:, None][:, None]), -1)
    return torch.matmul(U, V.transpose(1, 2))


def their_yaw(R: np.ndarray) -> float:
    """predict.py の DAD 全周ブランチと同一の yaw 抽出。"""
    ang = Rotation.from_matrix(R.T).as_euler("xyz", degrees=True)
    return limit_angle(ang[1])


@torch.no_grad()
def main() -> None:
    model = build_model()
    rows = [json.loads(l) for l in open(VAL / "val.jsonl")]
    print(f"val rows: {len(rows)}")

    preds: list[float] = []
    batch_size = 64
    for i in tqdm(range(0, len(rows), batch_size), desc="semiuhpe",
                  dynamic_ncols=True):
        chunk = rows[i:i + batch_size]
        xs = []
        for r in chunk:
            im = Image.open(VAL / r["image"]).convert("RGB").resize([224, 224])
            a = (np.asarray(im, np.float32) / 255.0 - MEAN) / STD
            xs.append(a.transpose(2, 0, 1))
        x = torch.from_numpy(np.stack(xs)).cuda()
        A = model(x)
        Rm = batch_A_to_R(A).numpy()
        preds += [their_yaw(Rm[j]) for j in range(len(chunk))]

    ours = np.array([signed(r["yaw_deg"]) for r in rows])
    theirs = np.array(preds)

    # 符号較正: operator 検証済み s001 行(最も信頼できるラベル)で実測
    trust = np.array([r.get("label_source") == "intent_operator_promoted"
                      for r in rows])
    def circ(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.abs((a - b + 180.0) % 360.0 - 180.0)
    cal = {s: float(circ(ours[trust], s * theirs[trust]).mean()) for s in (1, -1)}
    sign = min(cal, key=cal.get)
    print(f"符号較正(operator 検証済み {int(trust.sum())} 行): "
          f"+1 -> {cal[1]:.1f} deg, -1 -> {cal[-1]:.1f} deg => sign={sign:+d}")

    err = circ(ours, sign * theirs)
    yaws360 = np.array([r["yaw_deg"] % 360.0 for r in rows])
    per_bin = {f"{b:03d}-{b+30:03d}": round(float(err[(yaws360 >= b) & (yaws360 < b + 30)].mean()), 2)
               for b in range(0, 360, 30)
               if ((yaws360 >= b) & (yaws360 < b + 30)).any()}
    result = {
        "model": "SemiUHPE DAD-WildHead-EffNetV2-S (full-range)",
        "val_rows": len(rows),
        "sign": sign,
        "maae": round(float(err.mean()), 3),
        "median": round(float(np.median(err)), 3),
        "acc15": round(float((err <= 15).mean() * 100), 2),
        "acc30": round(float((err <= 30).mean() * 100), 2),
        "per_bin_mae": per_bin,
    }
    out = VAL / "eval_semiuhpe.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("saved:", out)


if __name__ == "__main__":
    main()
