import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import argparse
import os
import torch
from torch.utils.data import DataLoader
from utils.config import load_config
from datasets import BenchmarkSRDataset
from models import O2LIIFSR
from evaluators.sr_evaluator import evaluate_model, save_metrics
from utils.checkpoint import load_checkpoint

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/default_sr.yaml')
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', default='outputs/test_run')
    args = ap.parse_args()
    cfg = load_config(args.config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = O2LIIFSR(feat_ch=cfg['model']['encoder_channels'], n_blocks=cfg['model']['num_residual_blocks'], decoder_hidden=cfg['model']['decoder_hidden_dim']).to(device)
    load_checkpoint(args.ckpt, model, None, device)
    loader = DataLoader(BenchmarkSRDataset(cfg['dataset']['root'], 'test', cfg['dataset']['patch_size'], cfg['dataset']['query_points'], cfg['dataset']['scale_min'], cfg['dataset']['scale_max'], cfg['dataset']['interpolation_mode']), batch_size=1)
    metrics = evaluate_model(model, loader, device, cfg['eval']['angles'])
    os.makedirs(args.out, exist_ok=True)
    save_metrics(metrics, args.out)
    print(metrics)
