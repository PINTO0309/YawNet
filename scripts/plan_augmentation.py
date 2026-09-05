#!/usr/bin/env python3
"""yaw の合成データ追加生成量プランを JSON に出力する。

data/yawpose/labels_fixed.jsonl を 10° ビンで集計し、対象ビンについて
目標水準に対する不足数(= 追加生成推奨数)を算出する。

既定は全 36 ビン・目標 = 前方側(270°〜90°)ビンの中央値。
--ranges / --target で「難領域の重点補強」プランも作れる:
    uv run python scripts/plan_augmentation.py \
        --ranges 120-180,210-270 --target 1200 --out augmentation_plan_s005.json

出力: data/yawpose/<out>
  - bins[*].region: "rear"(90°〜270°)/ "front"(それ以外)
  - bins[*].generation_quota: 推奨水準の不足数に QA 落ち 15% を上乗せした生成指示数
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "yawpose"
BIN_DEG = 10
REAR_FROM = 90
REAR_TO = 270
OVERGEN = 1.15           # QA 落ち見込みの過剰生成率
SOURCES = ["synthetic_001", "synthetic_002", "synthetic_003", "synthetic_004", "synthetic_005", "synthetic_006"]


def parse_ranges(spec: str) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for part in spec.split(","):
        lo, hi = part.split("-")
        out.append((int(lo), int(hi)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranges", type=str, default="",
                    help='対象 yaw レンジ(例 "120-180,210-270")。省略時は全周')
    ap.add_argument("--target", type=int, default=0,
                    help="目標水準 [件/ビン]。省略時は前方側ビンの中央値")
    ap.add_argument("--out", type=str, default="augmentation_plan.json",
                    help="出力ファイル名(data/yawpose/ 配下)")
    args = ap.parse_args()
    ranges = parse_ranges(args.ranges) if args.ranges else None

    n_bins = 360 // BIN_DEG
    counts = {s: np.zeros(n_bins, dtype=int) for s in SOURCES}
    with open(OUT / "labels_fixed.jsonl") as f:
        for line in f:
            r = json.loads(line)
            counts[r["source"]][int(r["yaw_deg"] % 360.0 // BIN_DEG)] += 1
    total = sum(counts.values())
    edges = np.arange(0, 360, BIN_DEG)
    rear = (edges >= REAR_FROM) & (edges < REAR_TO)

    targets = {
        "median_front": int(np.median(total[~rear])),
        "mean_all": int(round(total.mean())),
        "max_bin": int(total.max()),
    }
    if args.target > 0:
        targets = {"specified": args.target, **targets}
        recommended = "specified"
    else:
        recommended = "median_front"
    rec_target = targets[recommended]

    def in_ranges(lo: int) -> bool:
        if ranges is None:
            return True
        return any(a <= lo < b for a, b in ranges)

    bins = []
    for i in range(n_bins):
        if not in_ranges(int(edges[i])):
            continue
        need = {name: int(max(0, t - total[i])) for name, t in targets.items()}
        bins.append({
            "yaw_from": int(edges[i]),
            "yaw_to": int(edges[i]) + BIN_DEG,
            "region": "rear" if rear[i] else "front",
            "current": int(total[i]),
            "current_by_source": {s: int(counts[s][i]) for s in SOURCES},
            "additional_needed": need,
            "generation_quota": int(math.ceil(need[recommended] * OVERGEN)),
        })

    def region_sum(region: str, key: str) -> int:
        return sum(b[key] if key == "generation_quota"
                   else b["additional_needed"][key]
                   for b in bins if b["region"] == region)

    plan = {
        "summary": {
            "bin_deg": BIN_DEG,
            "ranges": args.ranges or "all",
            "targets_per_bin": targets,
            "recommended_target": recommended,
            "overgen_factor": OVERGEN,
            "current_total": int(total.sum()),
            "additional_total": {
                name: int(sum(b["additional_needed"][name] for b in bins))
                for name in targets},
            "additional_by_region": {
                "rear_90_270": region_sum("rear", recommended),
                "front": region_sum("front", recommended),
            },
            "generation_quota_total": int(sum(b["generation_quota"] for b in bins)),
            "generation_quota_by_region": {
                "rear_90_270": region_sum("rear", "generation_quota"),
                "front": region_sum("front", "generation_quota"),
            },
            "notes": [
                "下限水準(推奨)は前方側(270°〜90°)ビンの中央値 = "
                f"{rec_target} 件/ビン。",
                "175-190度は現状 0 件(ソースデータに真後ろ±15度がほぼ存在しない)。",
                "前方の谷(20-70度・300-330度)は顔が見えるため sixd + ランドマークの"
                "自動符号検証が全面的に適用できる。",
                "目標はビン単位の下限であり、超過ビンの削減は含まない。",
                "generation_quota は QA 落ち見込み 15% を上乗せした生成指示数。",
            ],
            "source_file": "labels_fixed.jsonl",
        },
        "bins": bins,
    }
    path = OUT / args.out
    with open(path, "w") as f:
        json.dump(plan, f, indent=2, ensure_ascii=False)
    print(json.dumps(plan["summary"], indent=2, ensure_ascii=False))
    print(f"\n{'bin':>10} {'region':>6} {'現状':>6} {'不足':>6} {'指示数':>6}")
    for b in bins:
        if b["additional_needed"][recommended] > 0:
            print(f"{b['yaw_from']:>4}-{b['yaw_to']:<5} {b['region']:>6} "
                  f"{b['current']:>6} {b['additional_needed'][recommended]:>6} "
                  f"{b['generation_quota']:>6}")
    print("saved:", path)


if __name__ == "__main__":
    main()
