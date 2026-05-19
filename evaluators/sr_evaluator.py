import csv
import json
import os
import torch
from metrics.psnr_ssim import psnr, ssim_simple
from metrics.equivariance_metrics import rotation_equivariance_error, reflection_equivariance_error


def evaluate_model(model, loader, device, angles):
    model.eval()
    out = {'psnr': [], 'ssim': [], 'rot_ee': [], 'ref_ee': []}
    with torch.no_grad():
        for batch in loader:
            lr = batch['lr_image'].to(device)
            q = batch['query_coords'].to(device)
            gt = batch['gt_rgb'].to(device)
            cell = batch['cell'].to(device)
            scale = batch['scale'].to(device)
            pred = model(lr, q, cell, scale)
            out['psnr'].append(psnr(pred, gt))
            out['ssim'].append(ssim_simple(pred, gt))
            out['rot_ee'].append(rotation_equivariance_error(model, lr, q, cell, scale, angles))
            out['ref_ee'].append(reflection_equivariance_error(model, lr, q, cell, scale))
    return {k: float(sum(v)/max(1, len(v))) for k, v in out.items()}


def save_metrics(metrics_dict, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    with open(os.path.join(output_dir, 'metrics.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['metric', 'value'])
        for k, v in metrics_dict.items():
            w.writerow([k, v])
