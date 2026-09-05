#!/usr/bin/env python3
"""yaw ラベルの規約修正・符号検証(qa_sixd.jsonl の診断結果に基づく)。

診断(scripts/verify_labels_sixd.py, data/yawpose/qa_sixd.jsonl)で判明した事実:
  - s001 intent 系(operator 検証済み)は sixdrepnet360 と符号一致率 0.99、
    円周誤差中央値 11°で整合 → 基準規約として信頼できる。
  - s001 の label_source=sixdrepnet360 の行だけは符号が全件反転して保存されている
    (反転補正で円周誤差中央値 1.2°)→ 規約変換ミス。符号反転で修正する。
  - s002 / s003 の generation_plan yaw は規約自体が反転しており(|yaw| 30–150° で
    反転補正により中央値 111°→20°)、さらに約 3 割のサンプルで生成モデルが
    左右を取り違えている → 規約反転 + sixd によるサンプル単位の符号検証を行う。
  - s004 は yawpose 規約で生成済み(規約反転不要)。|yaw| 15〜95° のみ sixd で
    符号を照合し(不一致は flip)、sixd 不信頼時と後方帯(|yaw|>95°)は
    納品側 direction QA 通過済みの intent を維持する。

処理:
  1. crop_meta.jsonl + qa_sixd.jsonl + study_landmark_yaw.jsonl を突き合わせ、
     修正済みレコードを labels_fixed.jsonl に出力
     (yaw_deg は修正後、yaw_deg_orig / label_fix を併記)。
  2. 大角度サンプル(|yaw|>=AMBIG_MIN_DEG)の符号は sixd 信頼域では sixd で検証し、
     sixd 不信頼域では hrffa_vitl_ibug68 のランドマーク幾何(眼線方向へ射影した
     鼻先オフセット/眼間距離 → sin フィット)による概算 yaw の符号で検証する
     (適用条件: |yaw|<=95°、DEIMv2 Face(16) score>=0.3、|landmark_yaw|>=8°。
      フィージビリティ: kept サンプルで sixd 検証済み符号との一致 99.3%)。
     どちらでも検証できないものは除外。
  3. train/val(9:1、ソース x 10° ビン層化、seed 42)を再生成する。
"""
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from build_yawpose_dataset import OUT, stratified_split

QA_PATH = OUT / "qa_sixd.jsonl"
META_PATH = OUT / "crop_meta.jsonl"
STUDY_PATH = OUT / "study_landmark_yaw.jsonl"
FIXED_PATH = OUT / "labels_fixed.jsonl"

AMBIG_MIN_DEG = 15.0     # これ未満の |yaw| は符号の影響が小さく無検証で保持
SIXD_YAW_MIN = 10.0      # sixd の符号を信頼する最小 |sixd_yaw|
SIXD_PITCH_MAX = 60.0    # sixd を信頼する最大 |sixd_pitch|
LM_FACE_MIN = 0.3        # landmark 符号を使う最小 Face(16) スコア
LM_YAW_MIN = 8.0         # landmark 符号を信頼する最小 |landmark_yaw|
LM_MAX_ABS_YAW = 95.0    # landmark 補正の対象上限(これ超は準背面で対象外)


def signed(deg: float) -> float:
    d = deg % 360.0
    return d - 360.0 if d > 180.0 else d


def calibrate_landmark_yaw(study: list[dict[str, Any]]) -> tuple[float, float]:
    """s001 operator 検証済みサブセットで sin(yaw) = S*nose_ratio + B を当てはめる。"""
    cal = [r for r in study if r["group"] == "cal_s001"
           and abs(signed(r["yaw_label"])) <= 80 and r["face_score"] >= LM_FACE_MIN]
    ratio = np.array([r["nose_ratio"] for r in cal])
    target = np.sin(np.radians([signed(r["yaw_label"]) for r in cal]))
    a = np.stack([ratio, np.ones(len(cal))], axis=1)
    coef, _, _, _ = np.linalg.lstsq(a, target, rcond=None)
    return float(coef[0]), float(coef[1])


def main() -> None:
    qa: dict[str, dict[str, Any]] = {
        q["image"]: q for q in (json.loads(l) for l in open(QA_PATH))}
    rows: list[dict[str, Any]] = [json.loads(l) for l in open(META_PATH)]

    study_rows: list[dict[str, Any]] = [json.loads(l) for l in open(STUDY_PATH)]
    lm_s, lm_b = calibrate_landmark_yaw(study_rows)
    print(f"landmark calibration: sin(yaw) = {lm_s:.4f} * nose_ratio + {lm_b:.4f}")
    lm: dict[str, float | None] = {}   # image -> landmark yaw (deg) or None
    for r in study_rows:
        if r["face_score"] >= LM_FACE_MIN:
            v = max(-1.0, min(1.0, lm_s * r["nose_ratio"] + lm_b))
            lm[r["image"]] = math.degrees(math.asin(v))
        else:
            lm[r["image"]] = None

    fixed: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()
    for r in rows:
        q = qa.get(r["image"])
        src = r["source"]
        ls = r.get("label_source", "")
        yaw_orig = float(r["yaw_deg"])
        fix = "kept"
        yaw = yaw_orig

        if src == "synthetic_001":
            if ls == "sixdrepnet360":
                yaw = (-yaw_orig) % 360.0          # 保存時の符号反転を修正
                fix = "negated_storage_bug"
        elif src in ("synthetic_004", "synthetic_005", "synthetic_006"):
            # yawpose 規約で生成済み(規約反転不要)。極端ピッチが無く納品側の
            # direction QA も通過済みのため、|yaw| 15〜95° のみ sixd で符号を
            # 照合し(不一致は flip)、sixd 不信頼時と後方帯は intent を維持する
            tag = src.replace("synthetic_00", "s00")
            yaw0 = signed(yaw_orig)
            fix = f"kept_intent_{tag}"
            if AMBIG_MIN_DEG <= abs(yaw0) <= LM_MAX_ABS_YAW:
                sixd_ok = (q is not None
                           and abs(q["sixd_yaw"]) >= SIXD_YAW_MIN
                           and abs(q["sixd_pitch"]) < SIXD_PITCH_MAX)
                if sixd_ok:
                    if (q["sixd_yaw"] > 0) != (yaw0 > 0):
                        yaw0 = -yaw0
                        fix = f"{tag}+sixd_flip"
                    else:
                        fix = f"{tag}+sixd_confirm"
                else:
                    fix = f"{tag}_unverified_kept"
            yaw = yaw0 % 360.0
        else:  # synthetic_002 / synthetic_003
            yaw0 = -signed(yaw_orig)               # 生成プロンプト規約の反転を修正
            fix = "negated_convention"
            if abs(yaw0) >= AMBIG_MIN_DEG:
                sixd_ok = (q is not None
                           and abs(q["sixd_yaw"]) >= SIXD_YAW_MIN
                           and abs(q["sixd_pitch"]) < SIXD_PITCH_MAX)
                lm_yaw = lm.get(r["image"])
                lm_ok = (lm_yaw is not None
                         and abs(lm_yaw) >= LM_YAW_MIN
                         and abs(yaw0) <= LM_MAX_ABS_YAW)
                if sixd_ok:
                    if (q["sixd_yaw"] > 0) != (yaw0 > 0):
                        yaw0 = -yaw0               # 生成モデルの左右取り違えを修正
                        fix = "negated_convention+sixd_flip"
                elif lm_ok:
                    if (lm_yaw > 0) != (yaw0 > 0):
                        yaw0 = -yaw0
                        fix = "negated_convention+landmark_flip"
                    else:
                        fix = "negated_convention+landmark_confirm"
                else:
                    stats[f"{src}:dropped_ambiguous"] += 1
                    continue
            yaw = yaw0 % 360.0

        stats[f"{src}:{fix}"] += 1
        fixed.append({**r,
                      "yaw_deg": round(yaw, 4),
                      "yaw_deg_orig": yaw_orig,
                      "label_fix": fix})

    with open(FIXED_PATH, "w") as f:
        for r in fixed:
            f.write(json.dumps(r) + "\n")

    train, val = stratified_split(fixed)
    for split, split_rows in [("train", train), ("val", val)]:
        with open(OUT / f"{split}.jsonl", "w") as f:
            for r in split_rows:
                f.write(json.dumps({**r, "split": split}) + "\n")

    yaw_bins = Counter(int(r["yaw_deg"] // 30) * 30 for r in fixed)
    summary = {
        "total_in": len(rows),
        "total_out": len(fixed),
        "train": len(train),
        "val": len(val),
        "fixes": dict(sorted(stats.items())),
        "yaw_hist_30deg": {str(k): yaw_bins[k] for k in sorted(yaw_bins)},
        "thresholds": {"ambig_min_deg": AMBIG_MIN_DEG,
                       "sixd_yaw_min": SIXD_YAW_MIN,
                       "sixd_pitch_max": SIXD_PITCH_MAX,
                       "lm_face_min": LM_FACE_MIN,
                       "lm_yaw_min": LM_YAW_MIN,
                       "lm_max_abs_yaw": LM_MAX_ABS_YAW},
        "landmark_calibration": {"sin_slope": round(lm_s, 4),
                                 "sin_intercept": round(lm_b, 4)},
    }
    with open(OUT / "label_fix_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
