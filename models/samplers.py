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


class GroupLocalFeatureSampler(nn.Module):
    """Sample each group-indexed feature map at query coords.

    Input F_G: [B, G, C, H, W], coords: [B, N, 2]
    Output Z: [B, N, G, C+2(+2)(+1)]
    """
    def __init__(self, include_cell=True, include_scale=True):
        super().__init__()
        self.include_cell = include_cell
        self.include_scale = include_scale

    def forward(self, feat_g, coords, cell=None, scale=None):
        b, g, c, h, w = feat_g.shape
        n = coords.shape[1]
        fg = feat_g.view(b * g, c, h, w)
        grid = coords.unsqueeze(1).repeat(1, g, 1, 1).view(b * g, n, 1, 2)
        z = F.grid_sample(fg, grid, mode='bilinear', align_corners=False).squeeze(-1).transpose(1, 2).view(b, g, n, c).permute(0, 2, 1, 3)
        parts = [z, coords.unsqueeze(2).repeat(1, 1, g, 1)]
        if self.include_cell and cell is not None:
            parts.append(cell.unsqueeze(2).repeat(1, 1, g, 1))
        if self.include_scale and scale is not None:
            parts.append(scale.unsqueeze(1).unsqueeze(2).repeat(1, n, g, 1))
        return torch.cat(parts, dim=-1)
