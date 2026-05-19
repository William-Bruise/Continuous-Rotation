import math
import torch


def psnr(pred, gt):
    mse = torch.mean((pred - gt) ** 2).item()
    if mse <= 1e-12:
        return 99.0
    return -10 * math.log10(mse)


def ssim_simple(pred, gt):
    mu_x, mu_y = pred.mean(), gt.mean()
    var_x, var_y = pred.var(), gt.var()
    cov = ((pred - mu_x) * (gt - mu_y)).mean()
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return ((2 * mu_x * mu_y + c1) * (2 * cov + c2) / ((mu_x ** 2 + mu_y ** 2 + c1) * (var_x + var_y + c2))).item()
