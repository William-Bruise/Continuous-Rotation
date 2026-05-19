import math
import torch
from models.group_ops import rotate_coords, reflect_coords, rotate_image, reflect_image, rotate_coords_rad, rotate_image_rad


def rotation_consistency_loss(model, lr, coords, cell, scale, theta):
    pred = model(lr, coords, cell, scale)
    pred_rot = model(rotate_image(lr, theta), rotate_coords(coords, theta), cell, scale)
    return torch.mean(torch.abs(pred_rot - pred))


def reflection_consistency_loss(model, lr, coords, cell, scale, axis='x'):
    pred = model(lr, coords, cell, scale)
    pred_ref = model(reflect_image(lr, axis=axis), reflect_coords(coords, axis=axis), cell, scale)
    return torch.mean(torch.abs(pred_ref - pred))


def continuous_equivariance_loss(model, lr, coords, cell, scale, max_rotation_radians=2 * math.pi, interp_mode='bilinear'):
    theta = torch.rand(1, device=lr.device, dtype=lr.dtype) * max_rotation_radians
    y_theta = rotate_image_rad(lr, theta, mode=interp_mode)
    pred_a = model(y_theta, coords, cell, scale)
    coords_back = rotate_coords_rad(coords, -theta)
    pred_b = model(lr, coords_back, cell, scale)
    return torch.mean(torch.abs(pred_a - pred_b))
