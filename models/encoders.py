import torch.nn as nn


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
    """Placeholder for future strict O(2)-equivariant encoder."""
    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError('EquivariantEncoder is a planned interface placeholder.')
