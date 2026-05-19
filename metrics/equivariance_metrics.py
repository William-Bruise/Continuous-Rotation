import torch
from models.group_ops import rotate_image, rotate_coords, reflect_image, reflect_coords


def rotation_equivariance_error(model, lr, coords, cell, scale, angles):
    with torch.no_grad():
        base = model(lr, coords, cell, scale)
        denom = torch.mean(torch.abs(base)) + 1e-8
        errs = []
        for a in angles:
            pr = model(rotate_image(lr, a), rotate_coords(coords, a), cell, scale)
            errs.append((torch.mean(torch.abs(pr - base)) / denom).item())
        return float(sum(errs) / max(1, len(errs)))


def reflection_equivariance_error(model, lr, coords, cell, scale, axis='horizontal'):
    with torch.no_grad():
        base = model(lr, coords, cell, scale)
        pref = model(reflect_image(lr, axis), reflect_coords(coords, axis), cell, scale)
        denom = torch.mean(torch.abs(base)) + 1e-8
        return (torch.mean(torch.abs(pref - base)) / denom).item()
