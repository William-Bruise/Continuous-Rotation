import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalFeatureSampler(nn.Module):
    def __init__(self, neighborhood=2, include_cell=True, include_scale=True):
        super().__init__()
        self.neighborhood = neighborhood
        self.include_cell = include_cell
        self.include_scale = include_scale

    def forward(self, feat, coords, cell=None, scale=None):
        b, c, h, w = feat.shape
        n = coords.shape[1]
        grid = coords.view(b, n, 1, 2)
        f = F.grid_sample(feat, grid, mode='bilinear', align_corners=False).squeeze(-1).transpose(1, 2)
        parts = [f, coords]
        if self.include_cell and cell is not None:
            parts.append(cell)
        if self.include_scale and scale is not None:
            parts.append(scale.unsqueeze(1).repeat(1, n, 1))
        return torch.cat(parts, dim=-1)
