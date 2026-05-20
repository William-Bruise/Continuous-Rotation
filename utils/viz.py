import os
import torch
from torchvision.utils import save_image


def save_sr_triplet(lr, pred_hr, gt_hr, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    grid = torch.cat([lr, pred_hr, gt_hr], dim=-1)
    save_image(grid.clamp(0, 1), path)
