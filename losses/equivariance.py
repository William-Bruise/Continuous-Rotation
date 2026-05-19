import torch
from models.group_ops import rotate_coords, reflect_coords


def rotation_consistency_loss(model, lr, coords, cell, scale, theta):
    from models.group_ops import rotate_image
    pred = model(lr, coords, cell, scale)
    lr_rot = rotate_image(lr, theta)
    coords_rot = rotate_coords(coords, theta)
    pred_rot = model(lr_rot, coords_rot, cell, scale)
    return torch.mean(torch.abs(pred_rot - pred))


def reflection_consistency_loss(model, lr, coords, cell, scale, axis='horizontal'):
    from models.group_ops import reflect_image
    pred = model(lr, coords, cell, scale)
    lr_ref = reflect_image(lr, axis=axis)
    coords_ref = reflect_coords(coords, axis=axis)
    pred_ref = model(lr_ref, coords_ref, cell, scale)
    return torch.mean(torch.abs(pred_ref - pred))
