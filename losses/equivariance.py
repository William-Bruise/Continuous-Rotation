import torch
from models.group_ops import rotate_coords, reflect_coords, rotate_image, reflect_image


def rotation_consistency_loss(model, lr, coords, cell, scale, theta):
    pred = model(lr, coords, cell, scale)
    pred_rot = model(rotate_image(lr, theta), rotate_coords(coords, theta), cell, scale)
    return torch.mean(torch.abs(pred_rot - pred))


def reflection_consistency_loss(model, lr, coords, cell, scale, axis='x'):
    pred = model(lr, coords, cell, scale)
    pred_ref = model(reflect_image(lr, axis=axis), reflect_coords(coords, axis=axis), cell, scale)
    return torch.mean(torch.abs(pred_ref - pred))
