"""VRAM ティア別の学習プリセット。

設計方針:
  - 実効バッチ(micro_batch × grad_accum)はタスクごとに全ティアで同一とする。
    これにより学習ダイナミクス(lr スケジュール・勾配ノイズ)がマシンに依存せず、
    ティアをまたいだ --resume も安全になる(resume 整合チェックは実効バッチで行う)。
  - VRAM が小さいティアほど micro_batch を下げ grad_accum で補う。
  - dinov3_teacher の 8GB ティアのみ、optimizer 状態(AdamW で全パラメーターの
    約 12 byte/param)が載らないため後段ブロックのみ解凍する。

タスク:
  - yawnet_small : train_yawnet.py の size <= 128(実効 256)
  - yawnet_320   : train_yawnet.py の size >= 192(教師用、実効 128)
  - dinov3_teacher : train_teacher_dinov3.py(ViT-L 320px、実効 64)
  - distill      : distill_yawnet.py(実効 128)
"""
from __future__ import annotations

from dataclasses import dataclass

VRAM_TIERS = (8, 16, 96)


@dataclass(frozen=True)
class VramPreset:
    micro_batch: int
    grad_accum: int
    amp_dtype: str = "fp16"          # "fp16" | "bf16"
    trainable_blocks: int | None = None  # dinov3 のみ: None=全ブロック / N=後段 N ブロック
    workers: int = 16

    @property
    def effective_batch(self) -> int:
        return self.micro_batch * self.grad_accum


PRESETS: dict[tuple[str, int], VramPreset] = {
    # train_yawnet.py, size <= 128(実効 256)
    ("yawnet_small", 8):  VramPreset(256, 1),
    ("yawnet_small", 16): VramPreset(256, 1),
    ("yawnet_small", 96): VramPreset(256, 1),
    # train_yawnet.py, size >= 192(実効 128)
    ("yawnet_320", 8):  VramPreset(32, 4),
    ("yawnet_320", 16): VramPreset(64, 2),
    ("yawnet_320", 96): VramPreset(128, 1),
    # train_teacher_dinov3.py(ViT-L 320px、実効 64、bf16)
    ("dinov3_teacher", 8):  VramPreset(8, 8, amp_dtype="bf16", trainable_blocks=8),
    ("dinov3_teacher", 16): VramPreset(16, 4, amp_dtype="bf16", trainable_blocks=None),
    ("dinov3_teacher", 96): VramPreset(64, 1, amp_dtype="bf16", trainable_blocks=None),
    # distill_yawnet.py(実効 128。教師 forward は no_grad なので比較的軽い)
    ("distill", 8):  VramPreset(64, 2),
    ("distill", 16): VramPreset(128, 1),
    ("distill", 96): VramPreset(128, 1),
}


def get_preset(task: str, vram: int) -> VramPreset:
    key = (task, vram)
    if key not in PRESETS:
        raise KeyError(f"unknown preset: task={task} vram={vram} "
                       f"(vram は {VRAM_TIERS} のいずれか)")
    return PRESETS[key]
