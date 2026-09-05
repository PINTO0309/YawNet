# YawNet

Full-circle head-yaw regression via biternion (cos/sin) + von Mises κ, DINOv3 teacher-student distillation on a purely synthetic 42k dataset.

合成データによる 360° 頭部 yaw 推定(biternion: cos/sin 回帰)のデータセット構築・学習・蒸留・ONNX エクスポートのパイプライン。

## セットアップ(uv + venv)

パッケージバージョンは `pyproject.toml` / `uv.lock` で完全固定
([High-Angle_Robust_Fast_FaceAlignment](https://github.com/PINTO0309/High-Angle_Robust_Fast_FaceAlignment) と同一バージョンを採用)。

```bash
uv sync --frozen                                      # onnxruntime-gpu 1.22.0 (既定)
# TensorRT EP 検証時:
# uv sync --frozen --no-group ort --group tensorrt    # onnxruntime-gpu 1.26.0
uv run python scripts/build_yawpose_dataset.py
```

symlink を作れないボリューム(exFAT 等)上で作業する場合は venv をホーム側に逃がす:

```bash
export UV_PROJECT_ENVIRONMENT=$HOME/.venvs/yawnet   # シェルの rc に入れておくと楽
uv sync --frozen
uv run --no-sync python scripts/build_yawpose_dataset.py
```

## データ配置

このリポジトリには実データを含まない(`data/` は空)。以下を配置して使う:

- `data/synthetic_001..006/` — 元の合成データ(フル画像 + メタデータ)
- `data/yawpose/` — `scripts/build_yawpose_dataset.py` が生成する統合データセット
  (320x320 頭部クロップ、train:val = 9:1)。仕様・ラベル修正の詳細は
  [docs/yawpose_dataset.md](docs/yawpose_dataset.md)
- `models/` — 推論用 ONNX(頭部検出 `deimv2_wholebody49_boxes_only.onnx`、
  ラベル検証 `sixdrepnet360_1x3x224x224_full.onnx`、ランドマーク
  `hrffa_vitl_ibug68_1x3x320x320.onnx` ほか)
- `ckpts/` — DINOv3 事前学習重み(License 上コミットしない、実行時に参照)

## モデル一覧(パラメーター量と演算量)

演算量は ONNX graph の実測(`scripts/count_macs_onnx.py`、Conv/MatMul/Gemm の
積和を集計。GFLOPs = 2 × GMACs)。パラメーターは PyTorch 実装の値
(ONNX では BN 折り込みでわずかに減る)。

| Model | Role | Input | Params [M] | GMACs | GFLOPs |
|---|---|---|---:|---:|---:|
| YawNet-64 | Student | 1x3x64x64 | 0.77 | 0.013 | 0.026 |
| YawNet-96 | Student | 1x3x96x96 | 0.77 | 0.028 | 0.057 |
| YawNet-128 | Student | 1x3x128x128 | 0.77 | 0.050 | 0.101 |
| DINOv3 ViT-L + biternion head | Teacher | 1x3x320x320 | 304.20 | 130.684 | 261.367 |

## 学習

(以下 `uv run` は、venv を同期済みなら `uv run --no-sync` でも同じ)

### VRAM ティア(`--vram 8|16|96`)

バッチ関連の設定は `scripts/vram_presets.py` に分離されており、`--vram` で
マシンの VRAM ティアを指定するだけで解決される。**実効バッチ
(micro_batch × grad_accum)は全ティアで同一**なので、学習ダイナミクスは
マシンに依存せず、**ティアをまたいだ `--resume` も可能**(整合チェックは
実効バッチで行う)。

| タスク | 実効バッチ | 96GB | 16GB | 8GB |
|---|---:|---|---|---|
| yawnet(size ≤128) | 256 | 256×1 | 256×1 | 256×1 |
| yawnet(size ≥192) | 128 | 128×1 | 64×2 | 32×4 |
| DINOv3 教師 | 64 | 64×1(全ブロック) | 16×4(全ブロック) | 8×8(後段 8 ブロックのみ解凍) |
| 蒸留 | 128 | 128×1 | 128×1 | 64×2 |

### 直接学習(ベースライン、64/96/128)

```bash
uv run python scripts/train_yawnet.py \
--size 64

uv run python scripts/train_yawnet.py \
--size 96

uv run python scripts/train_yawnet.py \
--size 128
```

主な既定値: `--epochs 100 --batch 256 --lr 3e-3 --kappa 2.0 --width 1.0
--balance inv --balance-bin 10`。成果物は `runs/yawnet_<size>[_tag]/`
(`last.pt` / `best_{maae:.6f}.pt` / `train_log.jsonl` / `result.json`)。

### 教師学習(320x320、蒸留用)

```bash
uv run python scripts/train_yawnet.py \
--size 320 \
--width 2.0 \
--batch 32 \
--tag teacher
```

教師はデプロイしないため 2M パラメーター制約の対象外(width 2.0 で約 3.0M)。

### DINOv3 ViT-L 教師(320x320、最高精度狙い)

DINOv3 事前学習 backbone(304M)+ biternion ヘッド。重み・実装は HRFFA と同じ
方式で実行時に参照し、License 上リポジトリには含めない(教師はデプロイしない)。
入力正規化は ImageNet(学生系の center05 と異なる点は蒸留側が自動処理)。

```bash
uv run python scripts/train_teacher_dinov3.py \
--vram 96 \
--variant vitl16 \
--tag teacher

uv run python scripts/train_teacher_dinov3.py \
--vram 96 \
--variant vitl16 \
--tag teacher \
--resume
```

レシピ: epoch 0 は backbone 凍結 → 以後プリセット範囲を解凍、
差分 lr(backbone 2e-5 / head 2e-4)、bf16、grad_clip 1.0。
v5 から既定で **κ(確信度)ヘッド + von Mises NLL**(難サンプルを κ が自動減量、
`--no-kappa-head` で従来動作)と **EMA**(`--ema-decay 0.999`、0 で無効。
評価・best 保存は EMA 重み)を使用。κ の epoch 平均は train_log.jsonl の
`kappa_mean` に記録される。
lr スケジュールは `--lr-schedule cosine`(既定)/ `wsd`
(warmup → 一定 lr → 末尾 `--decay-epochs` で cosine 減衰 → 0。
一定区間は総 epoch 数に依存しないため、resume 時に `--epochs` を書き換えて
延長/短縮でき、`epochs = 現 epoch + decay_epochs` とすると即 decay に入る)。
成果物は `runs/dinov3_vitl16_320[_tag]/`(last.pt は optimizer 込みで約 2GB)。

### 蒸留(教師 → 学生)

```bash
uv run python scripts/distill_yawnet.py \
--teacher runs/dinov3_vitl16_320_teacher \
--student-size 96 \
--vram 8 \
--alpha 0.7 \
--beta 0.3
```

`--teacher` は run ディレクトリ(`best_*.pt` を自動発見)または `.pt` を指定。
教師の種類(YawNet / DINOv3)と入力正規化は checkpoint から自動判別される。
同条件ペア方式(幾何変換 + 劣化を 320 側で 1 回適用し、学生はその縮小版を見る。
解像度以外は完全に同条件)で
`loss = α·NLL(student, teacher) + β·NLL(student, GT)`(既定 α=0.7/β=0.3)。
成果物は `runs/yawnet_distill_<size>[_tag]/`。

### 中断からの再開

いずれのスクリプトも同一引数 + `--resume` で `last.pt` から
全状態(model / optimizer / scheduler / scaler / RNG)を完全復帰して再開:

```bash
uv run python scripts/train_yawnet.py \
--size 320 \
--width 2.0 \
--batch 32 \
--tag teacher \
--resume

uv run python scripts/distill_yawnet.py \
--teacher runs/yawnet_320_teacher \
--student-size 96 \
--resume
```

### 検証指標の見方

毎 epoch の検証(val 4,211 枚)で表示・記録される指標。誤差はすべて
**円周上の角度差**で計算する(例: 予測 359° と正解 1° の誤差は 2°。
358° の大外れとは扱わない)。

| 指標 | 単位 | 意味 | 見方・備考 |
|---|---|---|---|
| **maae** | 度 | 角度誤差の絶対値の平均(Mean Absolute Angular Error) | **モデル選抜の基準**(`best_*.pt` はこの値が最良の epoch)。0 に近いほど良い。ランダム予測で 90 |
| **med** | 度 | 角度誤差の中央値 | 大外れに引きずられない「典型的な誤差」。maae との乖離が大きい場合は少数の大外し(例: 背面での左右取り違え = 180° 級)がある |
| **acc15** | % | 誤差 15° 以内のサンプル割合 | 「実用上ほぼ正しい向きを指せた率」。100 に近いほど良い |
| **acc30** | % | 誤差 30° 以内のサンプル割合 | 同上(緩い基準) |
| **per_bin_mae** | 度 | 正解 yaw の 30° ビンごとの平均誤差 | `train_log.jsonl` / `result.json` のみ。正面(000-030 / 330-360)と背面(150-210)の精度差など**方向別の弱点**の確認に使う |

## ONNX エクスポート

```bash
uv run python scripts/export_onnx.py \
--ckpt runs/yawnet_distill_128_unified_v6u

# κ(確信度)を第 2 出力に含める場合:
uv run python scripts/export_onnx.py \
--ckpt runs/yawnet_distill_128_unified_v6u \
--with-kappa
```

教師/学生・κ ヘッドの有無は checkpoint から自動判別。
バッチ 1 でエクスポート → onnxslim(Gemm 融合なし)→ onnxsim → graph 正準化
→ torch vs ORT parity 検証 → N バッチ化(バッチ 1/2/3 一致検証)→ 監査、
まで一括実行し、`<run>/<stem>_{1,N}x3xSxS.onnx` を出力する。
入力契約: `images` (N,3,S,S)。学生は center05 正規化(x/127.5 − 1)、
教師は ImageNet 正規化。出力 `cos_sin` (N,2) は単位ベクトル、
`--with-kappa` 時は `kappa` (N)(von Mises 集中度 = 確信度)が加わる。

## 構成

### ドキュメント

- `docs/yawpose_dataset.md` — 統合データセット yawpose の仕様
  (クロップ規則、yaw 規約、ラベル修正・教師再ラベルの全記録、ファイル一覧)
- `docs/synthetic_004_generation_spec.md` — yaw 全周均衡化の追加生成指示書
- `docs/synthetic_005_generation_spec.md` — 難領域重点補強(120°〜180° / 210°〜270°)の追加生成指示書
- `docs/synthetic_006_generation_spec.md` — 90°〜120° 帯補強の追加生成指示書

### データセット構築・ラベル品質

- `scripts/build_yawpose_dataset.py` — データセット構築(DEIM CUDA 検出 → 5% マージン正方形クロップ → 320x320、再開可能)
- `scripts/verify_labels_sixd.py` — sixdrepnet360 による yaw ラベル全数検証(→ `qa_sixd.jsonl`)
- `scripts/fix_labels.py` — ラベル修正(符号規約の補正、sixd / ランドマークによる符号回収、検証不能行の除外)
- `scripts/study_landmark_yaw.py` — ランドマーク幾何による yaw 符号推定のフィージビリティ・キャリブレーション
- `scripts/relabel_rear_teacher.py` — 教師モデルによる s001 背面ラベルの監査・再ラベル(完全可逆)
- `scripts/plan_augmentation.py` — 分布均衡化の追加生成プラン算出
- `scripts/plot_yaw_distribution.py` / `plot_projected_distribution.py` / `plot_ypr_distribution.py` — 分布可視化

### モデル・学習

- `scripts/yawnet.py` — YawNet(MBConv+SE+SiLU、biternion 出力、約 0.77M パラメーター、κ ヘッド対応)
- `scripts/dinov3_yaw.py` — DINOv3 backbone + biternion ヘッドの教師モデル(HRFFA 方式の実行時ロード)
- `scripts/augment.py` — HRFFA D4 幾何拡張の yaw-only 適応版(全幾何変換を 1 つの 3x3 行列に合成して 1 回だけ warp。photometric / motion blur / random erase / low-res jitter を含む)
- `scripts/verify_cam_yaw_sign.py` — カメラ回転ワープが appearance 基準 yaw に与える係数の実測(CAM_YAW_COEF = 0.166)
- `scripts/yaw_dataset.py` — Dataset / バランスサンプラー / 蒸留用同条件ペア Dataset
- `scripts/ema.py` — EMA(指数移動平均)ヘルパー
- `scripts/vram_presets.py` — VRAM ティア(8/16/96GB)別のバッチ/蓄積/精度プリセット(実効バッチは全ティア共通)
- `scripts/train_yawnet.py` — 直接学習(64/96/128、von Mises NLL、AMP、balance sampler、`--unified`)
- `scripts/train_teacher_dinov3.py` — DINOv3 ViT-L 教師の学習(凍結→解凍、差分 lr、bf16、κ + EMA、wsd)
- `scripts/distill_yawnet.py` — 320x320 教師 → 低解像度学生の蒸留学習(checkpoint / resume / ログは train_yawnet.py と同一規約)
- `scripts/val_preview.py` / `render_preview.py` — 検証サンプルのリングダイアル可視化(best 更新時 / オフライン)

### エクスポート・評価

- `scripts/export_onnx.py` — ONNX エクスポート(最適化・parity・N バッチ化・監査まで一括)
- `scripts/count_macs_onnx.py` — ONNX graph からの MACs 実測
- `scripts/eval_semiuhpe.py` — SOTA 比較: SemiUHPE(arXiv 2404.02544)を yawpose val で全周 yaw 評価
