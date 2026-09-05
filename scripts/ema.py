"""指数移動平均(EMA)重み。学習時のみの部品で推論への影響はない。

- update() は optimizer 更新ごとに呼ぶ。序盤は decay をランプさせて
  初期重みへのバイアスを避ける(min(decay, (1+step)/(10+step)))。
- apply_to()/restore() で評価・保存時に EMA 重みへ一時的に入れ替える。
- state_dict()/load_state_dict() で resume に対応。
"""
from __future__ import annotations

import torch
from torch import nn


class Ema:
    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.step = 0
        self.shadow: dict[str, torch.Tensor] = {
            k: v.detach().clone().float()
            for k, v in model.state_dict().items()
            if v.dtype.is_floating_point}
        self._backup: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.step += 1
        d = min(self.decay, (1 + self.step) / (10 + self.step))
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].lerp_(v.float(), 1.0 - d)

    @torch.no_grad()
    def apply_to(self, model: nn.Module) -> None:
        self._backup = {k: v.detach().clone()
                        for k, v in model.state_dict().items()
                        if k in self.shadow}
        model.load_state_dict(
            {k: v.to(dtype=model.state_dict()[k].dtype) for k, v in self.shadow.items()},
            strict=False)

    @torch.no_grad()
    def restore(self, model: nn.Module) -> None:
        model.load_state_dict(self._backup, strict=False)
        self._backup = {}

    def state_dict(self) -> dict:
        return {"decay": self.decay, "step": self.step, "shadow": self.shadow}

    def load_state_dict(self, state: dict) -> None:
        self.decay = state["decay"]
        self.step = state["step"]
        self.shadow = state["shadow"]
