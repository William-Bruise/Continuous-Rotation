import random
import torch
import torch.nn.functional as F


def random_crop(img, size):
    _, h, w = img.shape
    if h < size or w < size:
        return F.interpolate(img.unsqueeze(0), size=(max(h, size), max(w, size)), mode='bilinear', align_corners=False).squeeze(0)
    top = random.randint(0, h - size)
    left = random.randint(0, w - size)
    return img[:, top:top+size, left:left+size]


def resize(img, size_hw, mode='bicubic'):
    return F.interpolate(img.unsqueeze(0), size=size_hw, mode=mode, align_corners=False).squeeze(0)
