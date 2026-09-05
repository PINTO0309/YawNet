"""YawNet: lightweight biternion yaw regressor (cos/sin output, L2-normalized).

Fully convolutional MBConv backbone with SE and SiLU, global average pooling,
so the same architecture serves 64x64 / 96x96 / 128x128 inputs.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SE(nn.Module):
    def __init__(self, ch: int, r: int = 4) -> None:
        super().__init__()
        hidden = max(8, ch // r)
        self.fc1 = nn.Conv2d(ch, hidden, 1)
        self.fc2 = nn.Conv2d(hidden, ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = F.adaptive_avg_pool2d(x, 1)
        s = F.silu(self.fc1(s))
        return x * torch.sigmoid(self.fc2(s))


class MBConv(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1, expand: int = 3,
                 use_se: bool = False) -> None:
        super().__init__()
        mid = cin * expand
        self.use_res = stride == 1 and cin == cout
        layers: list[nn.Module] = []
        if expand != 1:
            layers += [nn.Conv2d(cin, mid, 1, bias=False), nn.BatchNorm2d(mid), nn.SiLU(inplace=True)]
        layers += [
            nn.Conv2d(mid, mid, 3, stride, 1, groups=mid, bias=False),
            nn.BatchNorm2d(mid), nn.SiLU(inplace=True),
        ]
        if use_se:
            layers.append(SE(mid))
        layers += [nn.Conv2d(mid, cout, 1, bias=False), nn.BatchNorm2d(cout)]
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        return x + out if self.use_res else out


class YawNet(nn.Module):
    """Output: (N, 2) unit vector (cos(yaw), sin(yaw)).

    kappa_head=True では最終 FC が 3 出力になり、3 次元目が von Mises 集中度 κ
    (確信度)。forward() は従来どおり単位ベクトルのみ返し(後方互換)、
    κ が必要な場合は forward_with_kappa() を使う(教師 Dinov3YawNet と同一規約)。
    """

    def __init__(self, width: float = 1.0, dropout: float = 0.2,
                 kappa_head: bool = False) -> None:
        super().__init__()
        self.kappa_head = kappa_head
        def c(ch: int) -> int:
            return max(8, int(round(ch * width / 8)) * 8)

        self.stem = nn.Sequential(
            nn.Conv2d(3, c(24), 3, 2, 1, bias=False), nn.BatchNorm2d(c(24)), nn.SiLU(inplace=True))
        self.stages = nn.Sequential(
            MBConv(c(24), c(32), stride=2, expand=2),
            MBConv(c(32), c(32), stride=1, expand=2),
            MBConv(c(32), c(64), stride=2, expand=3),
            MBConv(c(64), c(64), stride=1, expand=3),
            MBConv(c(64), c(64), stride=1, expand=3),
            MBConv(c(64), c(96), stride=2, expand=3, use_se=True),
            MBConv(c(96), c(96), stride=1, expand=3, use_se=True),
            MBConv(c(96), c(96), stride=1, expand=3, use_se=True),
            MBConv(c(96), c(160), stride=2, expand=3, use_se=True),
            MBConv(c(160), c(160), stride=1, expand=3, use_se=True),
        )
        self.head_conv = nn.Sequential(
            nn.Conv2d(c(160), c(256), 1, bias=False), nn.BatchNorm2d(c(256)), nn.SiLU(inplace=True))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(c(256), 3 if kappa_head else 2)
        if kappa_head:
            # softplus(1.85) ≈ 2.0 = 従来の固定 kappa と同じ初期値から始める
            with torch.no_grad():
                self.fc.bias[2] = 1.85

    def _head_out(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stages(x)
        x = self.head_conv(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        x = self.dropout(x)
        return self.fc(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self._head_out(x)
        if self.kappa_head:
            out = out[:, :2]
        return F.normalize(out, dim=1, eps=1e-6)

    def forward_with_kappa(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(単位ベクトル, κ) を返す。κ は softplus で正値化し [1e-3, 100] にクリップ。"""
        out = self._head_out(x)
        unit = F.normalize(out[:, :2], dim=1, eps=1e-6)
        if not self.kappa_head:
            return unit, torch.full_like(out[:, 0], 2.0)
        kappa = F.softplus(out[:, 2]).clamp(1e-3, 100.0)
        return unit, kappa


def von_mises_loss(pred_unit: torch.Tensor, target_cos_sin: torch.Tensor,
                   kappa: float = 2.0) -> torch.Tensor:
    """Beyer et al. biternion von Mises loss: 1 - exp(kappa*(cos(delta)-1))."""
    cos_delta = (pred_unit * target_cos_sin).sum(dim=1)
    return (1.0 - torch.exp(kappa * (cos_delta - 1.0))).mean()


def von_mises_nll(pred_unit: torch.Tensor, kappa: torch.Tensor,
                  target_cos_sin: torch.Tensor) -> torch.Tensor:
    """κ 学習型の von Mises 負対数尤度(定数項 log 2π を除く)。

    NLL = -κ cosΔ + log I0(κ) = κ(1 - cosΔ) + log I0e(κ)
    (I0e は指数スケール済み Bessel。難サンプルは κ が下がり自動的に減量される)

    注意: 連続分布の NLL なので、予測が正確になり κ が上がる(密度が尖る)と
    **損失は負の値になる**。これは正常な挙動でありバグではない(ガウシアン NLL
    が σ 縮小で負になるのと同じ)。固定 κ 版 von_mises_loss とは定義が異なるため
    数値の直接比較はできず、品質比較は maae で行うこと。
    """
    cos_delta = (pred_unit * target_cos_sin).sum(dim=1)
    return (kappa * (1.0 - cos_delta)
            + torch.log(torch.special.i0e(kappa))).mean()


if __name__ == "__main__":
    m = YawNet()
    n = sum(p.numel() for p in m.parameters())
    print(f"params: {n:,}")
    for s in (64, 96, 128):
        y = m(torch.randn(2, 3, s, s))
        print(s, y.shape, y.norm(dim=1))
