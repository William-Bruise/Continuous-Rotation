import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import argparse
import csv
import json
import os
import torch
from torch.utils.data import DataLoader
from utils.config import load_config
from datasets import BenchmarkSRDataset
from models import O2LIIFSR, GroupO2LIIFSR, ContinuousGroupO2LIIFSR
from evaluators.sr_evaluator import evaluate_model
from utils.checkpoint import load_checkpoint
from trainers.sr_trainer import asisr_collate_fn


def build_model(cfg, device):
    mcfg = cfg['model']
    if mcfg.get('variant', 'baseline_liif') == 'continuous_so2':
        return ContinuousGroupO2LIIFSR(
            feat_ch=mcfg['encoder_channels'], n_blocks=mcfg['num_residual_blocks'], decoder_hidden=mcfg['decoder_hidden_dim'],
            num_fourier_orders=mcfg.get('num_fourier_orders', 3), num_orientation_quadrature=mcfg.get('num_orientation_quadrature', 8),
            orientation_quadrature_mode=mcfg.get('orientation_quadrature_mode', 'uniform'), max_rotation_radians=mcfg.get('max_rotation_radians', 6.283185307179586)
        ).to(device)
    if mcfg.get('variant', 'baseline_liif') in ('baseline_liif', 'baseline_liif_consistency'):
        return O2LIIFSR(feat_ch=mcfg['encoder_channels'], n_blocks=mcfg['num_residual_blocks'], decoder_hidden=mcfg['decoder_hidden_dim']).to(device)
    use_group_decoder = mcfg.get('variant') in ('group_encoder_group_decoder', 'group_encoder_group_decoder_consistency')
    return GroupO2LIIFSR(feat_ch=mcfg['encoder_channels'], n_blocks=mcfg['num_residual_blocks'], decoder_hidden=mcfg['decoder_hidden_dim'], K=mcfg.get('num_rotations', mcfg.get('group_K', 8)), pooling=mcfg.get('group_pooling', 'mean'), reflect_axis=mcfg.get('reflect_axis', 'x'), use_group_decoder=use_group_decoder, use_reflection=mcfg.get('use_reflection', True), num_angular_modes=mcfg.get('num_angular_modes', 3), num_spectral_blocks=mcfg.get('num_spectral_blocks', 2), spectral_residual=mcfg.get('spectral_residual', True), enable_spectral=mcfg.get('enable_spectral', True)).to(device)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/default_sr.yaml')
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', default='outputs/test_run')
    args = ap.parse_args()

    cfg = load_config(args.config)
    os.makedirs(args.out, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = build_model(cfg, device)
    load_checkpoint(args.ckpt, model, None, device)

    bench = cfg['benchmark_eval']
    datasets = bench['datasets']
    groups = {'in_scale': bench['in_scales'], 'out_scale': bench['out_scales']}

    rows = []
    nested = {}
    for mode, scales in groups.items():
        nested[mode] = {}
        for ds in datasets:
            nested[mode][ds] = {}
            for scale in scales:
                loader = DataLoader(BenchmarkSRDataset(cfg['dataset']['root'], dataset_name=ds, scale=scale, query_points=cfg['dataset']['query_points'], interp=cfg['dataset']['interpolation_mode']), batch_size=1, collate_fn=asisr_collate_fn)
                m = evaluate_model(model, loader, device, cfg['eval']['angles'])
                nested[mode][ds][f'x{scale}'] = {'psnr': m['psnr'], 'ssim': m['ssim']}
                rows.append({'mode': mode, 'dataset': ds, 'scale': f'x{scale}', 'psnr': m['psnr'], 'ssim': m['ssim'], 'rot_ee': m['rot_ee'], 'rot_ee_cont': m['rot_ee_cont'], 'ref_ee': m['ref_ee']})

    with open(os.path.join(args.out, 'benchmark_table.json'), 'w') as f:
        json.dump(nested, f, indent=2)
    with open(os.path.join(args.out, 'benchmark_table.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    print(json.dumps(nested, indent=2))
