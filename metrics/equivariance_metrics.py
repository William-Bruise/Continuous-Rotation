import math
import torch
from models.group_ops import rotate_image, rotate_coords, reflect_image, reflect_coords, rotate_image_rad, rotate_coords_rad


def rotation_equivariance_error(model, lr, coords, cell, scale, angles):
    with torch.no_grad():
        base = model(lr, coords, cell, scale)
        denom = torch.mean(torch.abs(base)) + 1e-8
        errs = []
        for a in angles:
            pr = model(rotate_image(lr, a), rotate_coords(coords, a), cell, scale)
            errs.append((torch.mean(torch.abs(pr - base)) / denom).item())
        return float(sum(errs) / max(1, len(errs)))


def continuous_rotation_equivariance_error(model, lr, coords, cell, scale, num_angles=8, max_rotation_radians=2 * math.pi):
    with torch.no_grad():
        errs = []
        for _ in range(num_angles):
            theta = torch.rand(1, device=lr.device, dtype=lr.dtype) * max_rotation_radians
            pa = model(rotate_image_rad(lr, theta), coords, cell, scale)
            pb = model(lr, rotate_coords_rad(coords, -theta), cell, scale)
            denom = torch.mean(torch.abs(pb)) + 1e-8
            errs.append((torch.mean(torch.abs(pa - pb)) / denom).item())
        return float(sum(errs) / max(1, len(errs)))


def reflection_equivariance_error(model, lr, coords, cell, scale, axis='x'):
    with torch.no_grad():
        base = model(lr, coords, cell, scale)
        pref = model(reflect_image(lr, axis), reflect_coords(coords, axis), cell, scale)
        denom = torch.mean(torch.abs(base)) + 1e-8
        return (torch.mean(torch.abs(pref - base)) / denom).item()
