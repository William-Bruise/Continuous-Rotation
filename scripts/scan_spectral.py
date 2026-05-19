import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import argparse
import copy
import csv
import json
import torch
from torch.utils.data import DataLoader
from utils.config import load_config
from trainers.sr_trainer import SRTrainer, asisr_collate_fn
from datasets import BenchmarkSRDataset
from evaluators.sr_evaluator import evaluate_model


def run_once(cfg):
    trainer = SRTrainer(cfg, exp_dir=cfg['train']['output_dir'], logger=type('L', (), {'info': print})())
    model = trainer.train()
    loader = DataLoader(BenchmarkSRDataset(cfg['dataset']['root'], 'test', cfg['dataset']['patch_size'], cfg['dataset']['query_points'], cfg['dataset']['scale_min'], cfg['dataset']['scale_max'], cfg['dataset']['interpolation_mode']), batch_size=1, collate_fn=asisr_collate_fn)
    return evaluate_model(model, loader, trainer.device, cfg['eval']['angles'])

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/default_sr.yaml')
    ap.add_argument('--out', default='outputs/spectral_scan.json')
    args = ap.parse_args()
    base = load_config(args.config)
    K_list = [4, 8, 12, 16]
    rows = []
    for K in K_list:
        for use_reflection in [False, True]:
            for M in range(1, K // 2 + 1):
                cfg = copy.deepcopy(base)
                cfg['model']['variant'] = 'group_encoder_group_decoder_consistency'
                cfg['model']['num_rotations'] = K
                cfg['model']['group_K'] = K
                cfg['model']['use_reflection'] = use_reflection
                cfg['model']['num_angular_modes'] = M
                cfg['train']['num_epochs'] = min(1, cfg['train']['num_epochs'])
                m = run_once(cfg)
                row = {'K': K, 'use_reflection': use_reflection, 'M': M, **m}
                rows.append(row)
                print(row)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(rows, f, indent=2)
    with open(args.out.replace('.json', '.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
