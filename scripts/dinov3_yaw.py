"""DINOv3 backbone + biternion ヘッドの yaw 教師モデル。

HRFFA(High-Angle_Robust_Fast_FaceAlignment)の方式を踏襲:
  - DINOv3 公式実装を torch.hub キャッシュ位置へ git clone し、実行時 import する
    (License 上コード・重みをこのリポジトリに含めない。教師はデプロイもしない)
  - 重みはローカルの .pth を直接指定(既定: HRFFA/ckpts の vitl16)
  - 入力正規化は ImageNet(DINOv3 hub backbone の要件)

ヘッド: CLS トークン + パッチ平均を連結 → MLP → 2 次元 → L2 正規化(biternion)。
set_trainable_blocks(k) で「後段 k ブロック + ヘッドのみ学習」に制限できる
(8GB ティアで AdamW の optimizer 状態を載せるための措置)。
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

_DINOV3_GIT = "https://github.com/facebookresearch/dinov3"

# variant -> (hub 関数名, 既定 ckpt ファイル名, 埋め込み次元)
DINOV3_VARIANTS: dict[str, tuple[str, str, int]] = {
    "vits16": ("dinov3_vits16", "dinov3_vits16_pretrain_lvd1689m-08c60483.pth", 384),
    "vitb16": ("dinov3_vitb16", "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth", 768),
    "vitl16": ("dinov3_vitl16", "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth", 1024),
}

# 重みの探索順: リポジトリ直下 ckpts/(gitignore 済み、他マシンへの持ち出し用)
# → HRFFA の ckpts/(このマシンのローカル環境)
CKPT_SEARCH_DIRS = (
    Path(__file__).resolve().parent.parent / "ckpts",
    Path("/home/b920405/git/High-Angle_Robust_Fast_FaceAlignment/ckpts"),
)


def find_ckpt(ckpt_name: str) -> Path:
    for d in CKPT_SEARCH_DIRS:
        if (d / ckpt_name).exists():
            return d / ckpt_name
    raise FileNotFoundError(
        f"DINOv3 重みが見つからない: {ckpt_name}(探索先: "
        f"{', '.join(str(d) for d in CKPT_SEARCH_DIRS)}。--dinov3-ckpt で明示指定可)")


def _ensure_hub_code() -> None:
    """DINOv3 公式実装を torch.hub キャッシュ位置に用意して import path に載せる。"""
    target = Path(torch.hub.get_dir()) / "facebookresearch_dinov3_main"
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", _DINOV3_GIT, str(target)],
                       check=True)
    if str(target) not in sys.path:
        sys.path.insert(0, str(target))


class Dinov3YawNet(nn.Module):
    """DINOv3 backbone + biternion yaw ヘッド。出力: (N, 2) 単位ベクトル。

    kappa_head=True(v5〜)では最終層が 3 出力になり、3 次元目が von Mises 集中度
    κ(確信度)。forward() は従来どおり単位ベクトルのみ返し(後方互換)、
    κ が必要な場合は forward_with_kappa() を使う。
    """

    def __init__(self, variant: str = "vitl16",
                 ckpt_path: str | Path | None = None,
                 head_hidden: int = 512, dropout: float = 0.1,
                 pretrained: bool = True,
                 kappa_head: bool = False) -> None:
        super().__init__()
        hub_fn, ckpt_name, dim = DINOV3_VARIANTS[variant]
        self.variant = variant
        self.embed_dim = dim
        self.kappa_head = kappa_head
        _ensure_hub_code()
        from dinov3.hub import backbones  # noqa: PLC0415
        if pretrained:
            path = Path(ckpt_path) if ckpt_path else find_ckpt(ckpt_name)
            if not path.exists():
                raise FileNotFoundError(f"DINOv3 重みが見つからない: {path}")
            self.backbone = getattr(backbones, hub_fn)(
                pretrained=True, weights=str(path))
        else:  # 蒸留側で state_dict をロードする場合(構造だけ作る)
            self.backbone = getattr(backbones, hub_fn)(pretrained=False)
        out_dim = 3 if kappa_head else 2
        self.head = nn.Sequential(
            nn.Linear(2 * dim, head_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(head_hidden, out_dim))
        if kappa_head:
            # softplus(1.85) ≈ 2.0 = 従来の固定 kappa と同じ初期値から始める
            with torch.no_grad():
                self.head[-1].bias[2] = 1.85

    def set_trainable_blocks(self, k: int | None) -> None:
        """backbone の学習対象を制御する。None=全ブロック / 0=全凍結 / N=後段 N ブロック。

        ヘッドは常に学習対象。norm(最終 LayerNorm)は k>0 なら学習対象に含める。
        """
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        if k is None:
            for p in self.backbone.parameters():
                p.requires_grad_(True)
        elif k > 0:
            blocks = self.backbone.blocks
            for blk in blocks[-k:]:
                for p in blk.parameters():
                    p.requires_grad_(True)
            if hasattr(self.backbone, "norm") and self.backbone.norm is not None:
                for p in self.backbone.norm.parameters():
                    p.requires_grad_(True)
        for p in self.head.parameters():
            p.requires_grad_(True)

    def backbone_parameters(self) -> list[nn.Parameter]:
        return list(self.backbone.parameters())

    def head_parameters(self) -> list[nn.Parameter]:
        return list(self.head.parameters())

    def _head_out(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone.get_intermediate_layers(
            x, n=1, reshape=True, return_class_token=True)
        patch, cls = feats[0]                       # (B,C,h,w), (B,C)
        feat = torch.cat([cls, patch.mean(dim=(2, 3))], dim=1)
        return self.head(feat)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self._head_out(x)
        return F.normalize(out[:, :2].float(), dim=1, eps=1e-6)

    def forward_with_kappa(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(単位ベクトル, κ) を返す。κ は softplus で正値化し [1e-3, 100] にクリップ。"""
        out = self._head_out(x)
        unit = F.normalize(out[:, :2].float(), dim=1, eps=1e-6)
        if not self.kappa_head:
            return unit, torch.full_like(out[:, 0], 2.0).float()
        kappa = F.softplus(out[:, 2].float()).clamp(1e-3, 100.0)
        return unit, kappa


if __name__ == "__main__":
    m = Dinov3YawNet("vitl16")
    n_all = sum(p.numel() for p in m.parameters())
    m.set_trainable_blocks(8)
    n_train = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"params total={n_all:,} trainable(last8+head)={n_train:,}")
    with torch.no_grad():
        y = m(torch.randn(1, 3, 320, 320))
    print("out:", y.shape, float(y.norm()))
