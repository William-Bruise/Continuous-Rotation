import math
import torch
import torch.nn.functional as F

ALIGN_CORNERS = False


def rotate_coords(p, theta):
    t = math.radians(theta)
    rot = torch.tensor([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]], device=p.device, dtype=p.dtype)
    return p @ rot.T


def reflect_coords(p, axis='x'):
    out = p.clone()
    if axis == 'x':  # M(x,y)=(x,-y)
        out[..., 1] = -out[..., 1]
    elif axis == 'y':
        out[..., 0] = -out[..., 0]
    else:
        raise ValueError(axis)
    return out


def apply_group_to_coords(coords, k, r, K=8, reflect_axis='x'):
    theta = 360.0 * (k % K) / K
    c = reflect_coords(coords, axis=reflect_axis) if int(r) == 1 else coords
    return rotate_coords(c, theta)


def apply_group_inverse_to_coords(coords, k, r, K=8, reflect_axis='x'):
    # T_g = R_theta M^r => T_g^{-1} = M^r R_{-theta}
    theta = -360.0 * (k % K) / K
    c = rotate_coords(coords, theta)
    return reflect_coords(c, axis=reflect_axis) if int(r) == 1 else c


def _warp_by_coord_map(x, map_fn, mode='bilinear'):
    b, _, h, w = x.shape
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, h, device=x.device, dtype=x.dtype),
        torch.linspace(-1, 1, w, device=x.device, dtype=x.dtype),
        indexing='ij',
    )
    base = torch.stack([xx, yy], dim=-1).unsqueeze(0).repeat(b, 1, 1, 1)
    src = map_fn(base)
    return F.grid_sample(x, src, mode=mode, padding_mode='border', align_corners=ALIGN_CORNERS)


def apply_group_to_image(x, k, r, K=8, reflect_axis='x', mode='bilinear'):
    # T_g y = warp(y, T_g^{-1})
    return _warp_by_coord_map(x, lambda c: apply_group_inverse_to_coords(c, k, r, K=K, reflect_axis=reflect_axis), mode=mode)


def rotate_image(x, theta, mode='bilinear'):
    # compatibility helper
    k = int(round(theta / 360.0 * 3600))
    return _warp_by_coord_map(x, lambda c: rotate_coords(c, -theta), mode=mode)


def reflect_image(x, axis='horizontal'):
    reflect_axis = 'x' if axis in ('horizontal', 'x') else 'y'
    return apply_group_to_image(x, k=0, r=1, K=1, reflect_axis=reflect_axis, mode='bilinear')


def inverse_group_action(kind, value):
    if kind == 'rotate':
        return ('rotate', -value)
    if kind == 'reflect':
        return ('reflect', value)
    raise ValueError(kind)


def rotate_coords_rad(p, theta_rad):
    ct = torch.cos(theta_rad)
    st = torch.sin(theta_rad)
    x = p[..., 0:1]
    y = p[..., 1:2]
    return torch.cat([ct * x - st * y, st * x + ct * y], dim=-1)


def rotate_image_rad(x, theta_rad, mode='bilinear'):
    # [T_theta x](u)=x(R_{-theta}u) via inverse map
    return _warp_by_coord_map(x, lambda c: rotate_coords_rad(c, -theta_rad), mode=mode)
