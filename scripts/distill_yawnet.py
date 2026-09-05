#!/usr/bin/env python3
"""320x320 教師 → 低解像度学生の蒸留学習。

方式(同条件ペア蒸留):
  - DistillYawDataset が幾何変換 + 劣化(photometric/lowres/blur/erase)を
    320 側で 1 回だけ適用した同一内容の 2 ビューを返す
    (教師 = 劣化済み 320、学生 = その縮小版。解像度以外は完全に同条件)
  - 教師は毎ステップ no_grad でオンライン推論(augment 済みビューを見るため
    事前計算はできない)
  - loss = alpha * vonMises(student, teacher出力) + beta * vonMises(student, GT)

教師 checkpoint は train_yawnet.py の runs/<teacher_run>/best_*.pt を指定する
(--teacher にはディレクトリまたは .pt ファイルのどちらでも可)。

チェックポイント(runs/<run_name>/)は train_yawnet.py と同一規約:
  last.pt(フル状態、--resume で完全復帰)/ best_{maae:.6f}.pt(最良のみ)/
  train_log.jsonl / result.json
"""
import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from yawnet import YawNet, von_mises_loss, von_mises_nll
from yaw_dataset import (DistillYawDataset, YawDataset, make_balanced_sampler)
from train_yawnet import epoch_print, evaluate, rng_states, restore_rng_states
from val_preview import render_val_preview, select_indices
from vram_presets import get_preset

ROOT = Path(__file__).resolve().parent.parent


def load_teacher(spec: str, device: str) -> tuple[torch.nn.Module, dict[str, Any]]:
    """--teacher で指定された run ディレクトリまたは .pt から教師を復元する。

    checkpoint の model_type から YawNet 教師 / DINOv3 教師を自動判別する。
    返す info の "norm" は教師ビューに要求される入力正規化。
    """
    path = Path(spec)
    if path.is_dir():
        cands = sorted(path.glob("best_*.pt"))
        if not cands:
            raise FileNotFoundError(f"{path} に best_*.pt が見つからない")
        path = cands[0]   # best は常に 1 つだけ残す運用
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model_type = ck.get("model_type", "yawnet")
    if model_type == "dinov3_yaw":
        from dinov3_yaw import Dinov3YawNet  # noqa: PLC0415
        kappa_head = ck["model"]["head.3.weight"].shape[0] == 3
        model: torch.nn.Module = Dinov3YawNet(ck["variant"], pretrained=False,
                                              kappa_head=kappa_head)
        norm = "imagenet"                    # DINOv3 の要件
        width = None
    else:
        width = float(ck["args"].get("width", 1.0))
        model = YawNet(width=width,
                       kappa_head=ck["model"]["fc.weight"].shape[0] == 3)
        norm = "center05"
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    info = {"path": str(path), "type": model_type, "width": width, "norm": norm,
            "size": ck["args"].get("size"), "metrics": ck.get("metrics")}
    return model, info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", type=str, required=True,
                    help="教師の run ディレクトリ(runs/yawnet_320_*)または best_*.pt")
    ap.add_argument("--student-size", type=int, default=96, choices=[64, 96, 128])
    ap.add_argument("--teacher-size", type=int, default=320)
    ap.add_argument("--width", type=float, default=1.0, help="学生の width")
    ap.add_argument("--alpha", type=float, default=0.7, help="蒸留損失の重み")
    ap.add_argument("--beta", type=float, default=0.3, help="GT 損失の重み")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--vram", type=int, default=8, choices=[8, 16, 96],
                    help="マシンの VRAM ティア(バッチ/蓄積は vram_presets.py で解決)")
    ap.add_argument("--batch", type=int, default=0,
                    help="micro batch の明示上書き(0 = プリセットに従う。蓄積は 1 になる)")
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--kappa", type=float, default=2.0)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--data", type=str, default=str(ROOT / "data" / "yawpose"))
    ap.add_argument("--balance", type=str, default="inv",
                    choices=["sqrt", "inv", "none"])
    ap.add_argument("--balance-bin", type=int, default=10)
    ap.add_argument("--balance-max-ratio", type=float, default=20.0)
    ap.add_argument("--kappa-head", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="学生の κ(確信度)ヘッド + von Mises NLL"
                         "(--no-kappa-head で従来の固定 κ 損失)")
    ap.add_argument("--unified", action="store_true",
                    help="train + val を統合して学習(意図的なデータリーク許容。"
                         "val 指標は選抜専用の参考値になる)")
    ap.add_argument("--resume", action="store_true",
                    help="runs/<run_name>/last.pt から全状態を復帰して再開")
    args = ap.parse_args()

    preset = get_preset("distill", args.vram)
    if args.batch > 0:
        micro_batch, grad_accum = args.batch, 1
    else:
        micro_batch, grad_accum = preset.micro_batch, preset.grad_accum
    effective_batch = micro_batch * grad_accum

    device = "cuda"
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    run_name = (f"yawnet_distill_{args.student_size:03d}"
                f"{'_unified' if args.unified else ''}"
                f"{('_' + args.tag) if args.tag else ''}")
    run_dir = ROOT / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    last_path = run_dir / "last.pt"

    teacher, teacher_info = load_teacher(args.teacher, device)
    print(f"teacher: {teacher_info['path']} (type={teacher_info['type']}, "
          f"norm={teacher_info['norm']}, trained@{teacher_info['size']}, "
          f"val={teacher_info['metrics'] and teacher_info['metrics'].get('maae')})")
    print(f"vram={args.vram}GB micro_batch={micro_batch} accum={grad_accum} "
          f"(effective {effective_batch})")

    data = Path(args.data)
    train_ds = DistillYawDataset(data, "unified" if args.unified else "train",
                                 args.student_size,
                                 teacher_size=args.teacher_size, train=True,
                                 teacher_norm=teacher_info["norm"])
    val_ds = YawDataset(data, "val", args.student_size, train=False)
    sampler = make_balanced_sampler(train_ds, bin_deg=args.balance_bin,
                                    mode=args.balance,
                                    max_ratio=args.balance_max_ratio)
    train_ld = DataLoader(train_ds, batch_size=micro_batch, sampler=sampler,
                          num_workers=args.workers, pin_memory=True,
                          drop_last=True, persistent_workers=True)
    val_batch = min(512, max(32, int(512 * (128 / args.student_size) ** 2)))
    val_ld = DataLoader(val_ds, batch_size=val_batch, shuffle=False,
                        num_workers=args.workers, pin_memory=True)

    student = YawNet(width=args.width, kappa_head=args.kappa_head).to(device)
    n_params = sum(p.numel() for p in student.parameters())
    print(f"run={run_name} student_params={n_params:,} "
          f"train={len(train_ds)} val={len(val_ds)}")

    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.wd)
    steps_per_epoch = len(train_ld) // grad_accum
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = args.warmup * steps_per_epoch

    def lr_at(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * p))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    scaler = torch.amp.GradScaler("cuda")

    start_epoch = 0
    best: dict[str, Any] = {"maae": 1e9}
    if args.resume:
        if not last_path.exists():
            raise FileNotFoundError(f"--resume 指定だが {last_path} が存在しない")
        ck = torch.load(last_path, map_location="cpu", weights_only=False)
        for k in ("student_size", "teacher_size", "width", "lr", "wd",
                  "kappa", "epochs", "warmup", "alpha", "beta", "unified",
                  "kappa_head", "balance", "balance_bin", "balance_max_ratio"):
            if getattr(args, k) != ck["args"][k]:
                raise ValueError(
                    f"resume 時のハイパーパラメーター不一致: {k} "
                    f"(checkpoint={ck['args'][k]}, now={getattr(args, k)})")
        ck_eff = ck.get("effective_batch", ck["args"].get("batch"))
        if ck_eff != effective_batch:
            raise ValueError(
                f"resume 時の実効バッチ不一致: checkpoint={ck_eff}, "
                f"now={effective_batch}(ティア間 resume は実効バッチが同じ前提)")
        student.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        restore_rng_states(ck["rng"])
        start_epoch = ck["epoch"] + 1
        best = ck["best"]
        print(f"resumed from {last_path} (next epoch {start_epoch}, "
              f"best maae {best['maae']})")

    log_path = run_dir / "train_log.jsonl"
    preview_indices = select_indices(val_ds)
    resumed = args.resume
    t0 = time.time()
    for epoch in range(start_epoch, args.epochs):
        student.train()
        run_total = run_kd = run_gt = 0.0
        kappa_sum = 0.0
        opt.zero_grad(set_to_none=True)
        bar = tqdm(train_ld, desc=f"ep {epoch}/{args.epochs - 1}",
                   dynamic_ncols=True, leave=False)
        for step, (t_x, s_x, target, _) in enumerate(bar, 1):
            t_x = t_x.to(device, non_blocking=True)
            s_x = s_x.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                t_pred = teacher(t_x)
            if args.kappa_head:
                with torch.autocast("cuda", dtype=torch.float16):
                    s_pred, s_kappa = student.forward_with_kappa(s_x)
                s_pred, s_kappa = s_pred.float(), s_kappa.float()
                loss_kd = von_mises_nll(s_pred, s_kappa, t_pred.float().detach())
                loss_gt = von_mises_nll(s_pred, s_kappa, target)
                kappa_sum += s_kappa.mean().item()
            else:
                with torch.autocast("cuda", dtype=torch.float16):
                    s_pred = student(s_x)
                loss_kd = von_mises_loss(s_pred.float(), t_pred.float().detach(),
                                         kappa=args.kappa)
                loss_gt = von_mises_loss(s_pred.float(), target, kappa=args.kappa)
            loss = args.alpha * loss_kd + args.beta * loss_gt
            scaler.scale(loss / grad_accum).backward()
            if step % grad_accum == 0:
                # fp16 較正で optimizer step がスキップされた回は scheduler も進めない
                prev_scale = scaler.get_scale()
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                if scaler.get_scale() >= prev_scale:
                    sched.step()
            run_total += loss.item()
            run_kd += loss_kd.item()
            run_gt += loss_gt.item()
            bar.set_postfix(loss=f"{run_total / step:.4f}",
                            kd=f"{run_kd / step:.4f}",
                            gt=f"{run_gt / step:.4f}",
                            lr=f"{opt.param_groups[0]['lr']:.2e}")

        metrics = evaluate(student, val_ld, device)
        improved = metrics["maae"] < best["maae"]
        if improved:
            best = {**metrics, "epoch": epoch}
            for old in run_dir.glob("best_*.pt"):
                old.unlink()
            torch.save({"model": student.state_dict(), "metrics": metrics,
                        "epoch": epoch, "args": vars(args), "params": n_params,
                        "teacher": teacher_info},
                       run_dir / f"best_{metrics['maae']:.6f}.pt")
            try:
                render_val_preview(student, val_ds, preview_indices,
                                   run_dir / "val_preview_best.png", device,
                                   epoch, metrics["maae"])
            except Exception as e:  # noqa: BLE001  プレビュー失敗で学習を止めない
                print(f"val preview failed: {e}")

        torch.save({
            "model": student.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(),
            "scaler": scaler.state_dict(),
            "rng": rng_states(),
            "epoch": epoch,
            "best": best,
            "metrics": metrics,
            "args": vars(args),
            "params": n_params,
            "effective_batch": effective_batch,
            "teacher": teacher_info,
        }, last_path)

        with open(log_path, "a") as f:
            f.write(json.dumps({
                "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "epoch": epoch,
                "loss": round(run_total / len(train_ld), 6),
                "loss_kd": round(run_kd / len(train_ld), 6),
                "loss_gt": round(run_gt / len(train_ld), 6),
                **({"kappa_mean": round(kappa_sum / len(train_ld), 3)}
                   if args.kappa_head else {}),
                "lr": opt.param_groups[0]["lr"],
                **metrics,
                "best_maae": best["maae"],
                "best_epoch": best.get("epoch"),
                "elapsed_sec": round(time.time() - t0, 1),
                **({"resumed": True} if resumed and epoch == start_epoch else {}),
            }) + "\n")

        epoch_print(
            f"ep {epoch:3d} loss {run_total/len(train_ld):.4f} "
            f"(kd {run_kd/len(train_ld):.4f} gt {run_gt/len(train_ld):.4f}) "
            + (f"kappa {kappa_sum/len(train_ld):5.1f} " if args.kappa_head else "")
            + f"maae {metrics['maae']:6.2f} med {metrics['median']:6.2f} "
            + f"acc15 {metrics['acc15']:5.1f} acc30 {metrics['acc30']:5.1f} "
            + f"(best {best['maae']:.2f}@{best.get('epoch','-')}) "
            + f"{time.time()-t0:6.0f}s", improved)

    with open(run_dir / "result.json", "w") as f:
        json.dump({"params": n_params, "best": best, "args": vars(args),
                   "teacher": teacher_info}, f, indent=2)
    print("BEST:", json.dumps(best))


if __name__ == "__main__":
    main()
