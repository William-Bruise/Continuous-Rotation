import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import argparse
import json

import torch

from utils.config import load_config
from models import O2LIIFSR, GroupO2LIIFSR, ContinuousGroupO2LIIFSR


def count_params(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def module_breakdown(model):
    out = {}
    for name, mod in model.named_children():
        out[name] = count_params(mod)
    return out


def build_model(cfg, device):
    mcfg = cfg['model']
    variant = mcfg.get('variant', 'continuous_so2')
    if variant == 'continuous_so2':
        return ContinuousGroupO2LIIFSR(
            feat_ch=mcfg['encoder_channels'],
            n_blocks=mcfg['num_residual_blocks'],
            decoder_hidden=mcfg['decoder_hidden_dim'],
            num_fourier_orders=mcfg.get('num_fourier_orders', 3),
            num_orientation_quadrature=mcfg.get('num_orientation_quadrature', 8),
            orientation_quadrature_mode=mcfg.get('orientation_quadrature_mode', 'uniform'),
            max_rotation_radians=mcfg.get('max_rotation_radians', 6.283185307179586),
        ).to(device)
    if variant in ('baseline_liif', 'baseline_liif_consistency'):
        return O2LIIFSR(
            feat_ch=mcfg['encoder_channels'],
            n_blocks=mcfg['num_residual_blocks'],
            decoder_hidden=mcfg['decoder_hidden_dim'],
        ).to(device)
    use_group_decoder = variant in ('group_encoder_group_decoder', 'group_encoder_group_decoder_consistency')
    return GroupO2LIIFSR(
        feat_ch=mcfg['encoder_channels'],
        n_blocks=mcfg['num_residual_blocks'],
        decoder_hidden=mcfg['decoder_hidden_dim'],
        K=mcfg.get('num_rotations', mcfg.get('group_K', 8)),
        pooling=mcfg.get('group_pooling', 'mean'),
        reflect_axis=mcfg.get('reflect_axis', 'x'),
        use_group_decoder=use_group_decoder,
        use_reflection=mcfg.get('use_reflection', True),
        num_angular_modes=mcfg.get('num_angular_modes', 3),
        num_spectral_blocks=mcfg.get('num_spectral_blocks', 2),
        spectral_residual=mcfg.get('spectral_residual', True),
        enable_spectral=mcfg.get('enable_spectral', True),
    ).to(device)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/default_sr.yaml')
    ap.add_argument('--out', default='outputs/param_count.json')
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = build_model(cfg, device)
    total = count_params(model)
    breakdown = module_breakdown(model)

    payload = {
        'variant': cfg['model'].get('variant', 'continuous_so2'),
        'total_params': total,
        'total_params_million': round(total / 1e6, 4),
        'module_params': breakdown,
        'module_params_million': {k: round(v / 1e6, 4) for k, v in breakdown.items()},
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)

    print(json.dumps(payload, indent=2))
