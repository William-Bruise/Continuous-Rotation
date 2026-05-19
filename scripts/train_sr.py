import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import argparse
import os
from datetime import datetime
from utils.config import load_config, save_config
from utils.logger import build_logger
from utils.seed import set_seed
from datasets.download_utils import ensure_dataset_structure
from trainers.sr_trainer import SRTrainer
from torch.utils.data import DataLoader
from datasets import BenchmarkSRDataset
from evaluators.sr_evaluator import evaluate_model, save_metrics

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/default_sr.yaml')
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg['seed'])
    ensure_dataset_structure(cfg['dataset']['root'])
    exp_dir = os.path.join(cfg['train']['output_dir'], datetime.utcnow().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(exp_dir, exist_ok=True)
    save_config(cfg, os.path.join(exp_dir, 'config.yaml'))
    logger = build_logger(os.path.join(exp_dir, 'train.log'))
    trainer = SRTrainer(cfg, exp_dir, logger)
    model = trainer.train()
    test_loader = DataLoader(BenchmarkSRDataset(cfg['dataset']['root'], 'test', cfg['dataset']['patch_size'], cfg['dataset']['query_points'], cfg['dataset']['scale_min'], cfg['dataset']['scale_max'], cfg['dataset']['interpolation_mode']), batch_size=1)
    metrics = evaluate_model(model, test_loader, trainer.device, cfg['eval']['angles'])
    save_metrics(metrics, exp_dir)
    logger.info(f'test metrics: {metrics}')
    print(exp_dir)
