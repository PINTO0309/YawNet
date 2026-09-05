#!/usr/bin/env python3
"""学習済み checkpoint から検証プレビュー(3x3)をオフライン生成する CLI。

学習スクリプトが best 更新時に自動出力するものと同一の画像を、任意の
checkpoint から生成する。教師(dinov3_yaw)/ 学生(YawNet)は checkpoint の
model_type から自動判別し、入力正規化・解像度も自動復元する
(ローダは export_onnx.py と共用)。

使い方:
    uv run python scripts/render_preview.py --ckpt runs/dinov3_vitl16_320_teacher_v3
    uv run python scripts/render_preview.py --ckpt runs/yawnet_distill_64_v3

出力は既定で <ckpt のフォルダ>/val_preview_best.png(--out で変更可)。
"""
import argparse
from pathlib import Path

import torch

from export_onnx import load_model
from val_preview import render_val_preview, select_indices
from yaw_dataset import YawDataset

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True,
                    help="run ディレクトリ(best_*.pt 自動発見)または .pt ファイル")
    ap.add_argument("--data", type=str, default=str(ROOT / "data" / "yawpose"))
    ap.add_argument("--out", type=str, default="",
                    help="出力 PNG(省略時は <ckpt のフォルダ>/val_preview_best.png)")
    args = ap.parse_args()

    model, ck_path, size, _, norm = load_model(args.ckpt, 0)
    device = "cuda"
    model.to(device)
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    epoch = int(ck.get("epoch", -1))
    maae = float((ck.get("metrics") or {}).get("maae", float("nan")))
    amp_dtype = torch.bfloat16 if norm == "imagenet" else torch.float16

    ds = YawDataset(args.data, "val", size, train=False, input_norm=norm)
    out = Path(args.out) if args.out else ck_path.parent / "val_preview_best.png"
    render_val_preview(model, ds, select_indices(ds), out, device,
                       epoch, maae, amp_dtype=amp_dtype)
    print(f"saved: {out} (ckpt={ck_path.name}, size={size}, norm={norm}, "
          f"epoch={epoch}, maae={maae})")


if __name__ == "__main__":
    main()
