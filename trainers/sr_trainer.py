import os
import random
import torch
from torch.utils.data import DataLoader
from datasets import DIV2KASISRDataset
from models import O2LIIFSR
from losses.reconstruction import reconstruction_l1
from losses.equivariance import rotation_consistency_loss, reflection_consistency_loss
from evaluators.sr_evaluator import evaluate_model, save_metrics
from utils.checkpoint import save_checkpoint, load_checkpoint


class SRTrainer:
    def __init__(self, cfg, exp_dir, logger):
        self.cfg, self.exp_dir, self.logger = cfg, exp_dir, logger
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = O2LIIFSR(feat_ch=cfg['model']['encoder_channels'], n_blocks=cfg['model']['num_residual_blocks'], decoder_hidden=cfg['model']['decoder_hidden_dim']).to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=cfg['train']['learning_rate'])
        self.start_epoch, self.best_psnr = 0, -1e9
        self._build_data()

    def _build_data(self):
        dcfg = self.cfg['dataset']
        self.train_loader = DataLoader(DIV2KASISRDataset(dcfg['root'], 'train', dcfg['patch_size'], dcfg['query_points'], dcfg['scale_min'], dcfg['scale_max'], dcfg['interpolation_mode']), batch_size=self.cfg['train']['batch_size'], shuffle=True)
        self.val_loader = DataLoader(DIV2KASISRDataset(dcfg['root'], 'val', dcfg['patch_size'], dcfg['query_points'], dcfg['scale_min'], dcfg['scale_max'], dcfg['interpolation_mode']), batch_size=1)

    def maybe_resume(self):
        p = os.path.join(self.exp_dir, 'checkpoints', 'latest.pt')
        if os.path.exists(p):
            ckpt = load_checkpoint(p, self.model, self.opt, self.device)
            self.start_epoch = ckpt['epoch'] + 1
            self.best_psnr = ckpt.get('best_metric', self.best_psnr)

    def train(self):
        self.maybe_resume()
        for epoch in range(self.start_epoch, self.cfg['train']['num_epochs']):
            self.model.train()
            for batch in self.train_loader:
                lr = batch['lr_image'].to(self.device)
                q = batch['query_coords'].to(self.device)
                gt = batch['gt_rgb'].to(self.device)
                cell = batch['cell'].to(self.device)
                scale = batch['scale'].to(self.device)
                pred = self.model(lr, q, cell, scale)
                loss = reconstruction_l1(pred, gt)
                if self.cfg['loss']['enable_rot']:
                    loss = loss + self.cfg['loss']['lambda_rot'] * rotation_consistency_loss(self.model, lr, q, cell, scale, random.choice(self.cfg['eval']['angles']))
                if self.cfg['loss']['enable_ref']:
                    loss = loss + self.cfg['loss']['lambda_ref'] * reflection_consistency_loss(self.model, lr, q, cell, scale)
                self.opt.zero_grad(); loss.backward(); self.opt.step()

            metrics = evaluate_model(self.model, self.val_loader, self.device, self.cfg['eval']['angles'])
            self.logger.info(f"epoch={epoch} metrics={metrics}")
            ckpt_dir = os.path.join(self.exp_dir, 'checkpoints')
            save_checkpoint(os.path.join(ckpt_dir, 'latest.pt'), self.model, self.opt, epoch, self.best_psnr)
            if metrics['psnr'] > self.best_psnr:
                self.best_psnr = metrics['psnr']
                save_checkpoint(os.path.join(ckpt_dir, 'best.pt'), self.model, self.opt, epoch, self.best_psnr)
            if (epoch + 1) % self.cfg['train']['validation_interval'] == 0:
                save_metrics(metrics, self.exp_dir)
        return self.model
