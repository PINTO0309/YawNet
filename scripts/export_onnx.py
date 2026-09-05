#!/usr/bin/env python3
"""YawNet モデル(DINOv3 教師 / YawNet 学生)の ONNX エクスポート CLI。

checkpoint の model_type から教師(dinov3_yaw)/ 学生(YawNet)を自動判別する。
検証パイプラインは onnx-export-optimize スキル(HRFFA ノウハウ)に準拠:
  torch export(バッチ 1、opset 17、TorchScript)
    → simplify_onnx.py(onnxslim [Gemm 融合なし] → onnxsim [定数畳み込みのみ、
       子プロセス + フォールバック] → 固定 graph 正準化 + rank-5 qkv の 4 次元化
       → ORT parity[graph 最適化 OFF])
    → torch vs ORT parity
    → fixed_to_nbatch.py(固定バッチ 1 → N、バッチ 1/2/3 一致検証込み)
    → audit_onnx.py(不変条件の機械チェック)

入力契約(いずれも `images` (N,3,S,S) → 出力 `cos_sin` (N,2) 単位ベクトル):
  - DINOv3 教師: ImageNet 正規化済み RGB(x/255 → (x - mean) / std)
  - YawNet 学生: center05 正規化済み RGB(x/127.5 - 1)

--with-kappa(κ ヘッド付き教師のみ): 出力を `cos_sin` (N,2) + `kappa` (N) の
2 本にする(κ = von Mises 集中度 = 確信度。ファイル名に _kappa を付与)。

使い方:
    uv run python scripts/export_onnx.py --ckpt runs/dinov3_vitl16_320_teacher_v3
    uv run python scripts/export_onnx.py --ckpt runs/yawnet_distill_64_v3

--ckpt はディレクトリ(best_*.pt を自動発見)または .pt ファイル。
出力は既定で「指定した重みと同じフォルダ」(--out-dir で変更可)。
ファイル名: 教師 = dinov3_<variant>_yaw_{1,N}x3xSxS.onnx(再エクスポートで上書き)、
学生 = <run 名>_{1,N}x3xSxS.onnx。
"""
import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
SKILL = Path.home() / ".claude" / "skills" / "onnx-export-optimize" / "scripts"


class KappaWrapper(torch.nn.Module):
    """(cos_sin, kappa) の 2 出力でエクスポートするための薄いラッパー。"""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model.forward_with_kappa(x)


def load_model(spec: str, size_override: int) -> tuple[torch.nn.Module, Path, int, str, str]:
    """checkpoint から (model, ckpt_path, size, stem, input_norm) を復元する。"""
    path = Path(spec)
    if path.is_dir():
        cands = sorted(path.glob("best_*.pt"))
        if not cands:
            raise FileNotFoundError(f"{path} に best_*.pt が見つからない")
        path = cands[0]
    ck: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=False)
    args = ck["args"]
    if ck.get("model_type") == "dinov3_yaw":
        from dinov3_yaw import Dinov3YawNet  # noqa: PLC0415
        model: torch.nn.Module = Dinov3YawNet(
            ck["variant"], pretrained=False,
            kappa_head=ck["model"]["head.3.weight"].shape[0] == 3)
        size = int(args.get("size", 320))
        stem = f"dinov3_{ck['variant']}_yaw"
        norm = "imagenet"
    else:
        from yawnet import YawNet  # noqa: PLC0415
        model = YawNet(width=float(args.get("width", 1.0)),
                       kappa_head=ck["model"]["fc.weight"].shape[0] == 3)
        size = int(args.get("student_size") or args.get("size"))
        stem = path.parent.name if path.parent != ROOT else path.stem
        norm = "center05"
    model.load_state_dict(ck["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, path, (size_override or size), stem, norm


def run(cmd: list[str]) -> None:
    print("$", " ".join(str(c) for c in cmd), flush=True)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise SystemExit(f"command failed (rc={r.returncode})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True,
                    help="run ディレクトリ(best_*.pt 自動発見)または .pt ファイル")
    ap.add_argument("--size", type=int, default=0,
                    help="入力解像度の上書き(0 = checkpoint の学習解像度)")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--out-dir", type=str, default="",
                    help="出力先(省略時は指定した重みと同じフォルダ)")
    ap.add_argument("--with-kappa", action="store_true",
                    help="κ(確信度)を第 2 出力に含める(κ ヘッド付き教師のみ)")
    args = ap.parse_args()

    model, ck_path, size, stem, norm = load_model(args.ckpt, args.size)
    n_params = sum(p.numel() for p in model.parameters())
    if args.with_kappa:
        if not getattr(model, "kappa_head", False):
            raise SystemExit("--with-kappa は κ ヘッド付き checkpoint のみ対応"
                             "(この checkpoint には κ 出力がない)")
        export_model: torch.nn.Module = KappaWrapper(model).eval()
        output_names = ["cos_sin", "kappa"]
        stem = f"{stem}_kappa"
    else:
        export_model = model
        output_names = ["cos_sin"]
    out_dir = Path(args.out_dir) if args.out_dir else ck_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = out_dir / f"{stem}_1x3x{size}x{size}_raw.onnx"
    fixed = out_dir / f"{stem}_1x3x{size}x{size}.onnx"
    nbatch = out_dir / f"{stem}_Nx3x{size}x{size}.onnx"

    print(f"ckpt: {ck_path} (params={n_params:,}, size={size}, input_norm={norm}, "
          f"outputs={output_names})")
    dummy = torch.randn(1, 3, size, size)
    torch.onnx.export(
        export_model, (dummy,), str(raw),
        opset_version=args.opset,
        input_names=["images"],
        output_names=output_names,
        dynamo=False,
    )
    print(f"raw export: {raw} ({raw.stat().st_size/1e6:.1f} MB)")

    # onnxslim → onnxsim → 正準化 → raw との ORT 一致検証(スキルの汎用スクリプト)
    run([sys.executable, str(SKILL / "simplify_onnx.py"), str(raw), str(fixed)])

    # torch vs ORT parity(graph 最適化 OFF)
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(str(fixed), so, providers=["CPUExecutionProvider"])
    with torch.no_grad():
        out = export_model(dummy)
    refs = [o.numpy() for o in (out if isinstance(out, tuple) else (out,))]
    gots = sess.run(None, {"images": dummy.numpy()})
    for name, ref, got in zip(output_names, refs, gots):
        tol = 5e-4 if name == "cos_sin" else 1e-3   # kappa は最大 100 の非正規化値
        err = float(np.abs(ref - got).max())
        print(f"parity {name}: torch vs ORT max_err={err:.2e} (tol {tol:.0e})")
        assert err < tol, f"parity failed for {name}: {err:.2e}"

    # 固定バッチ 1 → N バッチ化(バッチ 1/2/3 の数値一致検証込み)+ 監査
    run([sys.executable, str(SKILL / "fixed_to_nbatch.py"), str(fixed), str(nbatch)])
    run([sys.executable, str(SKILL / "audit_onnx.py"), str(fixed),
         "--pair", str(nbatch), "--check"])

    raw.unlink(missing_ok=True)
    print(f"\nartifacts:\n  fixed : {fixed} ({fixed.stat().st_size/1e6:.1f} MB)\n"
          f"  nbatch: {nbatch} ({nbatch.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
