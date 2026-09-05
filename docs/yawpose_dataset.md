# yawpose — unified dataset for 360° head-yaw estimation

Built by `scripts/build_yawpose_dataset.py`. Every sample is cropped from the
**original full frame** with DEIM head detection and saved at **320x320**.

## Yaw convention

- `yaw_deg` ∈ [0, 360). **0° = frontal (facing the camera), +90° = facing
  viewer-left** — the sixdrepnet360 yaw convention taken mod 360.
  The reference convention was confirmed against s001 operator-verified intent
  labels (sign agreement 0.99, median circular error 11° vs sixdrepnet360).
- The training target `(cos yaw, sin yaw)` is computed at load time.

## Sources

Counts are after label cleaning (`labels_fixed.jsonl`, 42,135 rows;
`images/` holds all 43,258 built crops — 1,123 s002/s003 crops whose yaw sign
could not be verified are excluded from the label files).

| Source | Rows | Yaw coverage | Label origin |
|---|---|---|---|
| synthetic_001 | 16,774 | 0–360° (full circle, incl. rear) | operator-verified intent 8,132 / sixdrepnet360 2,752 / rear intent 5,890 (2,655 of them teacher-relabeled) |
| synthetic_002 | 4,301 | ±98° (frontal, incl. extreme pitch ±118°) | generation plan (convention-corrected) |
| synthetic_003 | 2,576 | ±88° (upward pitch +2°…+118°) | generation plan (convention-corrected) |
| synthetic_004 | 12,062 | 20–329° (full-circle balancing batch) | generation intent (yawpose convention, no correction needed) |
| synthetic_005 | 5,710 | 120–269° (rear reinforcement) | generation intent |
| synthetic_006 | 712 | 100–119° (valley fill) | generation intent |

## Crop rule

`deim_long_side_square_5pct_per_side`:
head boxes from `models/deimv2_wholebody49_boxes_only.onnx` (CUDA, head =
label 7, score ≥ 0.3) are expanded by 5% per side, cropped as a long-side
square (shifted inward at image borders), and saved at 320x320.
Resampling uses OpenCV only (`INTER_AREA` when shrinking /
`INTER_LANCZOS4` when enlarging; PIL is avoided to reduce downsampling
error). When no head is detected the auto-QA head box is used as a fallback
(recorded in `box_source`; 2 samples); otherwise the frame is skipped and
logged.

## Label pipeline

### 1. Full-dataset verification (`scripts/verify_labels_sixd.py` → `qa_sixd.jsonl`)

Every crop is measured with sixdrepnet360. Its reliability window is
|sixd_yaw| ≥ 10° and |sixd_pitch| < 60°.

### 2. Corrections (`scripts/fix_labels.py` → `label_fix_summary.json`)

1. s001 rows with `label_source=sixdrepnet360` (2,752) had been stored with a
   flipped sign → **negated** (median circular error vs sixd drops to 1.2°).
2. s002/s003 generation-plan yaw used an inverted convention (median error
   111° → 20° after negation at |yaw| 30–150°) → **negated for all rows**.
3. ~15% of s002/s003 additionally had left/right swapped by the generator;
   within the sixd reliability window and |yaw| ≥ 15° the **sign is taken
   from sixd** (882 rows).
4. Outside the sixd window (extreme pitch etc.) the sign is recovered from
   **hrffa_vitl_ibug68 landmark geometry** (581 rows: 347 confirmed / 234
   flipped). Method: nose-tip offset projected on the eye line / inter-ocular
   distance → sine fit, calibrated on s001 verified labels in
   `scripts/study_landmark_yaw.py` (99.3% sign agreement with sixd-verified
   samples). Applied only for |yaw| ≤ 95°, DEIMv2 Face (classid 16)
   score ≥ 0.3, |landmark_yaw| ≥ 8°.
   1,123 rows verifiable by no method are **dropped**.
5. s004/s005/s006 are generated directly in the yawpose convention. sixd
   cross-check at |yaw| 15–95°: s004 2,626 rows with **0 sign mismatches**
   (sign agreement 0.999); rear ranges and sixd-unreliable rows keep the
   delivered intent (each batch passed the supplier-side direction QA).

Post-fix agreement with sixd: s002 sign 1.000 / median 11.2°, s003 1.000 /
8.4°, s001 0.923 / 13.2° (residual is sixd's own rear-view uncertainty).

### 3. Teacher relabeling of s001 rear labels (`scripts/relabel_rear_teacher.py` → `relabel_summary.json`)

The 5,890 s001 `intent_rear` rows (script-driven rig yaw, noisiest subset)
were audited with the DINOv3 teacher (v5, val MAAE 13.55°, hflip TTA).
Rows whose circular gap exceeded 25° were replaced with the teacher
prediction: **2,655 relabeled, 3,235 kept**. Split membership is unchanged;
every audited row keeps the pre-relabel value in `yaw_deg_orig`, the teacher
prediction in `teacher_yaw`, and the action in `label_fix` — so the step is
fully reversible from `labels_fixed.jsonl` alone.

## Split

`train.jsonl` 37,924 / `val.jsonl` 4,211 — a 9:1 stratified split
(source × 10° yaw bin, seed 42), regenerated from `labels_fixed.jsonl`.

## Files

- `images/` — 320x320 crops (`s001_`…`s006_` prefixes)
- `crop_meta.jsonl` — per-crop metadata incl. `head_box_xyxy` (build sidecar,
  source of truth for resumable builds)
- `labels_fixed.jsonl` — all cleaned records (`yaw_deg_orig` / `label_fix` /
  `teacher_yaw` audit fields included); train/val derive from it
- `label_fix_summary.json` / `relabel_summary.json` — cleaning and relabeling
  breakdowns
- `train.jsonl` / `val.jsonl` — the 9:1 split
- `qa_sixd.jsonl` — sixdrepnet360 verification for every crop
- `study_landmark_yaw.jsonl` — landmark-yaw feasibility study / calibration data
- `stats.json` — build-time aggregates (before label cleaning)
- `yaw_distribution.json` — yaw histogram (per bin / per source / per split)
- `yaw_hist_bar.png` / `yaw_hist_polar.png` — yaw distribution, 10° bins,
  stacked by source (`scripts/plot_yaw_distribution.py`)
- `ypr_hist.png` — yaw / pitch / roll distributions
  (`scripts/plot_ypr_distribution.py`; pitch from intent where available,
  sixd otherwise; roll from sixd)
- `augmentation_plan.json` (+ `_s005` / `_s006`) — full-circle balancing plans
  (`scripts/plan_augmentation.py`)
- `yaw_hist_bar_projected.png` — projected distribution after planned
  additions (`scripts/plot_projected_distribution.py`)
- `eval_semiuhpe.json` / `eval_semiuhpe_azimuth.json` /
  `eval_semiuhpe_table.png` — SOTA comparison: SemiUHPE
  (arXiv 2404.02544) evaluated full-range on our val set
  (`scripts/eval_semiuhpe.py`; azimuth variant = fair facing-vector yaw)
