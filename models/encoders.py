import torch
import torch.nn as nn
from .group_ops import apply_group_to_image


class ResidualBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(c, c, 3, 1, 1), nn.ReLU(inplace=True), nn.Conv2d(c, c, 3, 1, 1))

    def forward(self, x):
        return x + self.body(x)


class BaseEncoder(nn.Module):
    def __init__(self, in_ch=3, feat_ch=64, n_blocks=8):
        super().__init__()
        self.head = nn.Conv2d(in_ch, feat_ch, 3, 1, 1)
        self.blocks = nn.Sequential(*[ResidualBlock(feat_ch) for _ in range(n_blocks)])
        self.tail = nn.Conv2d(feat_ch, feat_ch, 3, 1, 1)

    def forward(self, x):
        f = self.head(x)
        return self.tail(self.blocks(f)) + f


class EquivariantEncoder(nn.Module):
    """Lifting-based discrete O(2) approximation (NOT strict group convolution).

    Enc_G(y) = { E_base(T_g y) }_{g in G}, output shape [B, |G|, C, H, W].
    Future work: replace with true O(2)-equivariant group convolution encoder.
    """
    def __init__(self, in_ch=3, feat_ch=64, n_blocks=8, K=8, reflect_axis='x'):
        super().__init__()
        self.base = BaseEncoder(in_ch=in_ch, feat_ch=feat_ch, n_blocks=n_blocks)
        self.K = K
        self.reflect_axis = reflect_axis
        self.group_elements = [(k, r) for r in (0, 1) for k in range(K)]

    def forward(self, x):
        feats = []
        for k, r in self.group_elements:
            xg = apply_group_to_image(x, k=k, r=r, K=self.K, reflect_axis=self.reflect_axis)
            feats.append(self.base(xg).unsqueeze(1))
        return torch.cat(feats, dim=1)
