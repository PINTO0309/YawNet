#!/usr/bin/env python3
"""DINOv3 ViT 教師(320x320、biternion yaw)の学習。

VRAM ティア(--vram 8|16|96)ごとの設定は scripts/vram_presets.py に分離:
  - 実効バッチ(micro_batch × grad_accum = 64)は全ティア共通。ティア間で
    学習ダイナミクスが変わらず、ティアをまたいだ --resume も可能
    (resume 整合チェックは実効バッチで行う)
  - 8GB ティアのみ backbone 後段 8 ブロック + ヘッドだけを学習
    (AdamW の optimizer 状態を載せるため)。16/96GB は全ブロック学習

レシピ(HRFFA 教師に準拠):
  - epoch 0 は backbone 凍結(ヘッドのみ)、以後プリセットの範囲を解凍
  - 差分 lr: backbone 2e-5 / head 2e-4、bf16、grad_clip 1.0
  - 入力正規化は ImageNet(DINOv3 の要件。学生系の center05 とは異なる)

lr スケジュール(--lr-schedule):
  - cosine(既定): warmup → cosine 減衰で終端 ≈ 0。総 epoch 数に依存するため
    resume 時に epochs は変更不可。
  - wsd: warmup → 一定 lr(ピーク)→ 末尾 --decay-epochs で cosine 減衰 → 0。
    一定区間は総 epoch 数に依存しないため、resume 時に --epochs を書き換えて
    延長/短縮できる(epochs = 現 epoch + decay_epochs とすると即 decay へ入る)。
    長さはいずれも optimizer step 単位で決まり、VRAM ティア間で一致する。

checkpoint(runs/<run_name>/)は train_yawnet.py と同一規約。ただし last.pt は
optimizer 状態込みで数 GB になる点に注意。
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

from dinov3_yaw import Dinov3YawNet
from ema import Ema
from yawnet import von_mises_loss, von_mises_nll
from yaw_dataset import YawDataset, make_balanced_sampler, angular_error_deg
from train_yawnet import epoch_print, rng_states, restore_rng_states
from val_preview import render_val_preview, select_indices
from vram_presets import get_preset

ROOT = Path(__file__).resolve().parent.parent


def make_lr_fn(schedule: str, warmup_steps: int, total_steps: int,
               decay_steps: int):
    """lr 倍率関数(optimizer step 単位)を返す。

    cosine: warmup 後、残り全区間で 1 → 0 の cosine 減衰。
    wsd   : warmup 後は 1.0 で一定、最後の decay_steps で 1 → 0 の cosine 減衰
            (decay_steps=0 は減衰なし = warmup + stable のみ)。
    """
    def lr_at(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        if schedule == "cosine":
            p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))
        # wsd
        decay_start = total_steps - decay_steps
        if step < decay_start or decay_steps == 0:
            return 1.0
        p = (step - decay_start) / max(1, decay_steps)
        return 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))
    return lr_at


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: str,
             amp_dtype: torch.dtype) -> dict[str, Any]:
    model.eval()
    errs: list[torch.Tensor] = []
    yaws: list[torch.Tensor] = []
    for x, _, yaw in tqdm(loader, desc="val", dynamic_ncols=True, leave=False):
        with torch.autocast("cuda", dtype=amp_dtype):
            pred = model(x.to(device, non_blocking=True))
        errs.append(angular_error_deg(pred.float().cpu(), yaw))
        yaws.append(yaw)
    errs_t = torch.cat(errs)
    yaws_t = torch.cat(yaws)
    per_bin: dict[str, float] = {}
    for b in range(0, 360, 30):
        m = (yaws_t >= b) & (yaws_t < b + 30)
        if m.any():
            per_bin[f"{b:03d}-{b+30:03d}"] = round(errs_t[m].mean().item(), 2)
    return {
        "maae": round(errs_t.mean().item(), 3),
        "median": round(errs_t.median().item(), 3),
        "acc15": round((errs_t <= 15).float().mean().item() * 100, 2),
        "acc30": round((errs_t <= 30).float().mean().item() * 100, 2),
        "per_bin_mae": per_bin,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vram", type=int, default=8, choices=[8, 16, 96],
                    help="マシンの VRAM ティア(バッチ等は vram_presets.py で解決)")
    ap.add_argument("--variant", type=str, default="vitl16",
                    choices=["vits16", "vitb16", "vitl16"])
    ap.add_argument("--dinov3-ckpt", type=str, default="",
                    help="DINOv3 事前学習重み .pth(省略時は HRFFA/ckpts の既定)")
    ap.add_argument("--init-ckpt", type=str, default="",
                    help="既存 run の best_*.pt(またはその run ディレクトリ)から"
                         "初期化する warm start。省略時は DINOv3 事前学習から")
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr-backbone", type=float, default=2e-5)
    ap.add_argument("--lr-head", type=float, default=2e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--kappa", type=float, default=2.0)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--lr-schedule", type=str, default="cosine",
                    choices=["cosine", "wsd"],
                    help="wsd = warmup → 一定 lr → 末尾 decay-epochs で cosine 減衰")
    ap.add_argument("--decay-epochs", type=int, default=5,
                    help="wsd の減衰区間(epoch 数)。cosine では無視される")
    ap.add_argument("--freeze-epochs", type=int, default=1,
                    help="この epoch 数だけ backbone を全凍結(ヘッドのみ学習)")
    ap.add_argument("--kappa-head", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="κ(確信度)ヘッド + von Mises NLL(v5〜の既定。"
                         "--no-kappa-head で従来の固定 κ 損失)")
    ap.add_argument("--ema-decay", type=float, default=0.999,
                    help="EMA の減衰率(0 で無効)。評価・best 保存は EMA 重みで行う")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--data", type=str, default=str(ROOT / "data" / "yawpose"))
    ap.add_argument("--balance", type=str, default="inv",
                    choices=["sqrt", "inv", "none"])
    ap.add_argument("--balance-bin", type=int, default=10)
    ap.add_argument("--balance-max-ratio", type=float, default=20.0)
    ap.add_argument("--unified", action="store_true",
                    help="train + val を統合して学習(意図的なデータリーク許容。"
                         "val 指標は選抜専用の参考値になる)")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    preset = get_preset("dinov3_teacher", args.vram)
    amp_dtype = torch.bfloat16 if preset.amp_dtype == "bf16" else torch.float16
    device = "cuda"
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    run_name = (f"dinov3_{args.variant}_{args.size:03d}"
                f"{'_unified' if args.unified else ''}"
                f"{('_' + args.tag) if args.tag else ''}")
    run_dir = ROOT / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    last_path = run_dir / "last.pt"

    data = Path(args.data)
    train_ds = YawDataset(data, "unified" if args.unified else "train",
                          args.size, train=True, input_norm="imagenet")
    val_ds = YawDataset(data, "val", args.size, train=False,
                        input_norm="imagenet")
    sampler = make_balanced_sampler(train_ds, bin_deg=args.balance_bin,
                                    mode=args.balance,
                                    max_ratio=args.balance_max_ratio)
    train_ld = DataLoader(train_ds, batch_size=preset.micro_batch, sampler=sampler,
                          num_workers=preset.workers, pin_memory=True,
                          drop_last=True, persistent_workers=True)
    val_ld = DataLoader(val_ds, batch_size=max(16, preset.micro_batch),
                        shuffle=False, num_workers=preset.workers, pin_memory=True)

    if args.init_ckpt:
        init_path = Path(args.init_ckpt)
        if init_path.is_dir():
            cands = sorted(init_path.glob("best_*.pt"))
            if not cands:
                raise FileNotFoundError(f"{init_path} に best_*.pt が見つからない")
            init_path = cands[0]
        ick = torch.load(init_path, map_location="cpu", weights_only=False)
        if ick.get("model_type") != "dinov3_yaw" or ick.get("variant") != args.variant:
            raise ValueError(
                f"--init-ckpt は同 variant の dinov3_yaw checkpoint のみ: "
                f"type={ick.get('model_type')}, variant={ick.get('variant')}")
        model = Dinov3YawNet(args.variant, pretrained=False,
                             kappa_head=args.kappa_head)
        sd = ick["model"]
        old_rows = sd["head.3.weight"].shape[0]
        new_rows = model.head[-1].weight.shape[0]
        if old_rows != new_rows:
            # κ 無し ckpt → κ 有りモデル(またはその逆): 共通行のみ移植
            with torch.no_grad():
                n = min(old_rows, new_rows)
                model.head[-1].weight[:n] = sd["head.3.weight"][:n]
                model.head[-1].bias[:n] = sd["head.3.bias"][:n]
            sd = {k: v for k, v in sd.items()
                  if k not in ("head.3.weight", "head.3.bias")}
            model.load_state_dict(sd, strict=False)
            print(f"warm start: head 出力 {old_rows}->{new_rows} 行(共通 {n} 行を移植)")
        else:
            model.load_state_dict(sd)
        model.to(device)
        print(f"warm start: {init_path} (val_maae="
              f"{(ick.get('metrics') or {}).get('maae')})")
    else:
        model = Dinov3YawNet(args.variant,
                             ckpt_path=args.dinov3_ckpt or None,
                             kappa_head=args.kappa_head).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    model.set_trainable_blocks(preset.trainable_blocks)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"run={run_name} vram={args.vram}GB micro_batch={preset.micro_batch} "
          f"accum={preset.grad_accum} (effective {preset.effective_batch}) "
          f"amp={preset.amp_dtype}")
    print(f"params total={n_params:,} trainable={n_train:,} "
          f"(blocks={'all' if preset.trainable_blocks is None else preset.trainable_blocks}) "
          f"train={len(train_ds)} val={len(val_ds)}")

    opt = torch.optim.AdamW([
        {"params": model.backbone_parameters(), "lr": args.lr_backbone},
        {"params": model.head_parameters(), "lr": args.lr_head},
    ], weight_decay=args.wd)
    if args.lr_schedule == "wsd" and not (0 <= args.decay_epochs <= args.epochs):
        raise ValueError(f"--decay-epochs は 0..epochs の範囲: {args.decay_epochs}")
    opt_steps_per_epoch = len(train_ld) // preset.grad_accum
    total_steps = args.epochs * opt_steps_per_epoch
    warmup_steps = args.warmup * opt_steps_per_epoch
    decay_steps = args.decay_epochs * opt_steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, make_lr_fn(args.lr_schedule, warmup_steps, total_steps, decay_steps))
    scaler = torch.amp.GradScaler("cuda", enabled=(preset.amp_dtype == "fp16"))
    ema = Ema(model, decay=args.ema_decay) if args.ema_decay > 0 else None

    start_epoch = 0
    best: dict[str, Any] = {"maae": 1e9}
    if args.resume:
        if not last_path.exists():
            raise FileNotFoundError(f"--resume 指定だが {last_path} が存在しない")
        ck = torch.load(last_path, map_location="cpu", weights_only=False)
        # wsd は一定区間が総 epoch 数に依存しないため、resume 時の epochs /
        # decay_epochs の変更(延長・短縮・即 decay 入り)を許可する
        check_keys = ["variant", "size", "lr_backbone", "lr_head", "wd",
                      "kappa", "warmup", "freeze_epochs", "lr_schedule",
                      "kappa_head", "ema_decay", "unified",
                      "balance", "balance_bin", "balance_max_ratio"]
        if args.lr_schedule == "cosine":
            check_keys += ["epochs", "decay_epochs"]
        for k in check_keys:
            if getattr(args, k) != ck["args"][k]:
                raise ValueError(
                    f"resume 時のハイパーパラメーター不一致: {k} "
                    f"(checkpoint={ck['args'][k]}, now={getattr(args, k)})")
        if ck["effective_batch"] != preset.effective_batch:
            raise ValueError(
                f"resume 時の実効バッチ不一致: checkpoint={ck['effective_batch']}, "
                f"now={preset.effective_batch}(ティア間 resume は実効バッチが同じ前提)")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        if ema is not None and "ema" in ck:
            ema.load_state_dict(ck["ema"])
            for v in ema.shadow.values():
                v.data = v.data.to(device)
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
        # 凍結スケジュール: freeze_epochs 未満はヘッドのみ、以後プリセット範囲を解凍
        if epoch < args.freeze_epochs:
            model.set_trainable_blocks(0)
        else:
            model.set_trainable_blocks(preset.trainable_blocks)

        model.train()
        running = 0.0
        kappa_sum = 0.0
        opt.zero_grad(set_to_none=True)
        bar = tqdm(train_ld, desc=f"ep {epoch}/{args.epochs - 1}",
                   dynamic_ncols=True, leave=False)
        for step, (x, target, _) in enumerate(bar, 1):
            x = x.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            if args.kappa_head:
                with torch.autocast("cuda", dtype=amp_dtype):
                    pred, kappa = model.forward_with_kappa(x)
                loss = von_mises_nll(pred.float(), kappa.float(), target)
                kappa_sum += kappa.mean().item()
            else:
                with torch.autocast("cuda", dtype=amp_dtype):
                    pred = model(x)
                loss = von_mises_loss(pred.float(), target, kappa=args.kappa)
            scaler.scale(loss / preset.grad_accum).backward()
            running += loss.item()
            if step % preset.grad_accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(
                    (p for p in model.parameters() if p.requires_grad),
                    args.grad_clip)
                # fp16 較正で optimizer step がスキップされた回は scheduler も進めない
                # (bf16 では scaler 無効のためスキップは起きず、常に進む)
                prev_scale = scaler.get_scale()
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                if scaler.get_scale() >= prev_scale:
                    sched.step()
                if ema is not None:
                    ema.update(model)
            bar.set_postfix(loss=f"{running / step:.4f}",
                            lr=f"{opt.param_groups[1]['lr']:.2e}")

        # 評価・best 保存は EMA 重みで行う(有効時)
        if ema is not None:
            ema.apply_to(model)
        metrics = evaluate(model, val_ld, device, amp_dtype)
        improved = metrics["maae"] < best["maae"]
        if improved:
            best = {**metrics, "epoch": epoch}
            for old in run_dir.glob("best_*.pt"):
                old.unlink()
            # ema.apply_to() 適用中なので state_dict は EMA 重み(有効時)
            torch.save({"model": model.state_dict(), "metrics": metrics,
                        "epoch": epoch, "args": vars(args), "params": n_params,
                        "model_type": "dinov3_yaw", "variant": args.variant,
                        "kappa_head": args.kappa_head,
                        "weights": "ema" if ema is not None else "raw"},
                       run_dir / f"best_{metrics['maae']:.6f}.pt")
            try:
                render_val_preview(model, val_ds, preview_indices,
                                   run_dir / "val_preview_best.png", device,
                                   epoch, metrics["maae"], amp_dtype=amp_dtype)
            except Exception as e:  # noqa: BLE001  プレビュー失敗で学習を止めない
                print(f"val preview failed: {e}")

        if ema is not None:
            ema.restore(model)

        torch.save({
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(),
            "scaler": scaler.state_dict(),
            "rng": rng_states(),
            "epoch": epoch,
            "best": best,
            "metrics": metrics,
            "args": vars(args),
            "params": n_params,
            "effective_batch": preset.effective_batch,
            "model_type": "dinov3_yaw",
            "variant": args.variant,
            "kappa_head": args.kappa_head,
            **({"ema": ema.state_dict()} if ema is not None else {}),
        }, last_path)

        with open(log_path, "a") as f:
            f.write(json.dumps({
                "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "epoch": epoch,
                "loss": round(running / len(train_ld), 6),
                "lr_head": opt.param_groups[1]["lr"],
                **({"kappa_mean": round(kappa_sum / len(train_ld), 3)}
                   if args.kappa_head else {}),
                **metrics,
                "best_maae": best["maae"],
                "best_epoch": best.get("epoch"),
                "elapsed_sec": round(time.time() - t0, 1),
                **({"resumed": True} if resumed and epoch == start_epoch else {}),
            }) + "\n")

        epoch_print(
            f"ep {epoch:3d} loss {running/len(train_ld):.4f} "
            + (f"kappa {kappa_sum/len(train_ld):5.1f} " if args.kappa_head else "")
            + f"maae {metrics['maae']:6.2f} med {metrics['median']:6.2f} "
            f"acc15 {metrics['acc15']:5.1f} acc30 {metrics['acc30']:5.1f} "
            f"(best {best['maae']:.2f}@{best.get('epoch','-')}) "
            f"{time.time()-t0:6.0f}s", improved)

    with open(run_dir / "result.json", "w") as f:
        json.dump({"params": n_params, "best": best, "args": vars(args)}, f, indent=2)
    print("BEST:", json.dumps(best))


if __name__ == "__main__":
    main()
