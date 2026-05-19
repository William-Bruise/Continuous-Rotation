import math
import torch
import torch.nn.functional as F


def _affine(theta_deg, b, device):
    t = math.radians(theta_deg)
    c, s = math.cos(t), math.sin(t)
    a = torch.tensor([[c, -s, 0.0], [s, c, 0.0]], device=device).unsqueeze(0).repeat(b, 1, 1)
    return a


def rotate_image(x, theta, mode='bilinear'):
    b = x.shape[0]
    grid = F.affine_grid(_affine(theta, b, x.device), x.size(), align_corners=False)
    return F.grid_sample(x, grid, mode=mode, padding_mode='border', align_corners=False)


def reflect_image(x, axis='horizontal'):
    if axis == 'horizontal':
        return torch.flip(x, dims=[3])
    return torch.flip(x, dims=[2])


def rotate_coords(p, theta):
    t = math.radians(theta)
    rot = torch.tensor([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]], device=p.device, dtype=p.dtype)
    return p @ rot.T


def reflect_coords(p, axis='horizontal'):
    out = p.clone()
    if axis == 'horizontal':
        out[..., 0] = -out[..., 0]
    else:
        out[..., 1] = -out[..., 1]
    return out


def inverse_group_action(kind, value):
    if kind == 'rotate':
        return ('rotate', -value)
    if kind == 'reflect':
        return ('reflect', value)
    raise ValueError(kind)
