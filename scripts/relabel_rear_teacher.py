#!/usr/bin/env python3
"""s001 intent_rear ラベルを DINOv3 教師で再ラベルする。

背景(教師学習ログの分析より):
  val の 240-270° ビンだけ MAE が突出(45° vs 鏡像ビン 19°)。学習は hflip で
  左右対称化されるため、非対称誤差はモデルではなく s001 intent_rear ラベルの
  片側ノイズを示す(該当帯の label vs sixd 乖離中央値 52° vs 鏡像帯 33°)。

方式:
  - 対象: labels_fixed.jsonl の source=synthetic_001 かつ label_source=intent_rear
    (train / val 両方。教師予測は全対象行に記録する)
  - 教師: runs/<teacher_run>/best_*.pt(train_teacher_dinov3.py の成果物)を
    distill と同じローダで復元。入力は ImageNet 正規化・320x320
  - hflip TTA: pred(x) と mirror(pred(flip(x))) の単位ベクトル平均
    (モデルの左右対称性を利用してノイズを均す)
  - 置換規則: 教師予測とラベルの円周差 > --thr(既定 25°)の行のみ
    yaw_deg を教師予測に置換(label_fix="teacher_relabel")。閾値内は intent を維持
    (label_fix に "+teacher_ok" を付記)
  - train/val の所属は一切変更しない(split を引き直すと教師の学習画像が val へ
    移動して自己成就評価になるため)。labels_fixed / train / val の 3 ファイルを
    画像キーで突き合わせて更新する

出力: 上記 3 jsonl の更新(元ファイルは .bak_relabel に退避)+
      relabel_summary.json
"""
import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from tqdm import tqdm

from distill_yawnet import load_teacher
from yaw_dataset import normalize_image

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "yawpose"


def signed(deg: float) -> float:
    d = deg % 360.0
    return d - 360.0 if d > 180.0 else d


def circ_diff(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


@torch.no_grad()
def predict_yaws(teacher: torch.nn.Module, norm: str, images: list[str],
                 size: int, batch: int, device: str) -> dict[str, float]:
    """hflip TTA 付きで教師の yaw [deg, 0..360) を推論する。"""
    preds: dict[str, float] = {}
    for i in tqdm(range(0, len(images), batch), desc="teacher", dynamic_ncols=True):
        chunk = images[i:i + batch]
        xs, xs_f = [], []
        for rel in chunk:
            bgr = cv2.imread(str(OUT / rel), cv2.IMREAD_COLOR)
            im = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if im.shape[0] != size:
                im = cv2.resize(im, (size, size), interpolation=cv2.INTER_LINEAR)
            xs.append(normalize_image(im, norm))
            xs_f.append(normalize_image(cv2.flip(im, 1), norm))
        x = torch.stack(xs).to(device)
        xf = torch.stack(xs_f).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = teacher(x).float()
            pf = teacher(xf).float()
        # 鏡像の予測は yaw -> -yaw なので sin 成分の符号を戻して平均
        pf[:, 1] = -pf[:, 1]
        v = torch.nn.functional.normalize(p + pf, dim=1)
        yaw = torch.rad2deg(torch.atan2(v[:, 1], v[:, 0])) % 360.0
        for rel, yv in zip(chunk, yaw.tolist()):
            preds[rel] = round(yv, 4)
    return preds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", type=str,
                    default=str(ROOT / "runs" / "dinov3_vitl16_320_teacher"))
    ap.add_argument("--thr", type=float, default=25.0,
                    help="この円周差 [deg] を超えた行のみ教師予測で置換")
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    device = "cuda"
    teacher, info = load_teacher(args.teacher, device)
    print(f"teacher: {info['path']} (type={info['type']}, val_maae="
          f"{info['metrics'] and info['metrics'].get('maae')})")

    fixed = [json.loads(l) for l in open(OUT / "labels_fixed.jsonl")]
    targets = [r for r in fixed
               if r["source"] == "synthetic_001"
               and r.get("label_source") == "intent_rear"]
    print(f"対象 (s001 intent_rear): {len(targets)}")

    preds = predict_yaws(teacher, info["norm"], [r["image"] for r in targets],
                         args.size, args.batch, device)

    # 更新テーブル: image -> (new_yaw, fix_tag, teacher_yaw)
    updates: dict[str, tuple[float, str, float]] = {}
    n_relabel = 0
    diffs: list[float] = []
    for r in targets:
        t_yaw = preds[r["image"]]
        d = circ_diff(r["yaw_deg"], t_yaw)
        diffs.append(d)
        if d > args.thr:
            updates[r["image"]] = (t_yaw, "teacher_relabel", t_yaw)
            n_relabel += 1
        else:
            updates[r["image"]] = (r["yaw_deg"], r["label_fix"] + "+teacher_ok", t_yaw)

    def apply(path: Path) -> int:
        rows = [json.loads(l) for l in open(path)]
        n = 0
        for r in rows:
            u = updates.get(r["image"])
            if u is None:
                continue
            new_yaw, tag, t_yaw = u
            if "yaw_deg_orig" not in r:
                r["yaw_deg_orig"] = r["yaw_deg"]
            r["yaw_deg"] = new_yaw
            r["label_fix"] = tag
            r["teacher_yaw"] = t_yaw
            n += 1
        shutil.copy2(path, path.with_suffix(path.suffix + ".bak_relabel"))
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        return n

    counts = {p.name: apply(p) for p in
              [OUT / "labels_fixed.jsonl", OUT / "train.jsonl", OUT / "val.jsonl"]}

    d = np.array(diffs)
    summary: dict[str, Any] = {
        "teacher": info["path"],
        "teacher_val_maae": info["metrics"] and info["metrics"].get("maae"),
        "threshold_deg": args.thr,
        "targets": len(targets),
        "relabeled": n_relabel,
        "kept": len(targets) - n_relabel,
        "diff_teacher_vs_label": {
            "mean": round(float(d.mean()), 2),
            "median": round(float(np.median(d)), 2),
            "p90": round(float(np.percentile(d, 90)), 2),
        },
        "updated_rows": counts,
        "note": "train/val の所属は不変(ラベルのみ更新)。旧値は yaw_deg_orig、"
                "教師予測は teacher_yaw に全対象行で記録。",
    }
    with open(OUT / "relabel_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
