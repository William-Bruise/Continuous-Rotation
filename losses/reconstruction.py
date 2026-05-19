import torch


def reconstruction_l1(pred, gt):
    return torch.mean(torch.abs(pred - gt))
