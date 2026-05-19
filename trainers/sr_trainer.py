import os
import random
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import DIV2KASISRDataset
from models import O2LIIFSR
from losses.reconstruction import reconstruction_l1
from losses.equivariance import rotation_consistency_loss, reflection_consistency_loss
from evaluators.sr_evaluator import evaluate_model, save_metrics
from utils.checkpoint import save_checkpoint, load_checkpoint


def _pad_to_max_hw(images):
    max_h = max(x.shape[-2] for x in images)
    max_w = max(x.shape[-1] for x in images)
    out = []
    for x in images:
        pad_h = max_h - x.shape[-2]
        pad_w = max_w - x.shape[-1]
        out.append(F.pad(x, (0, pad_w, 0, pad_h), mode='replicate'))
    return torch.stack(out, dim=0)


def asisr_collate_fn(batch):
    collated = {}
    collated['lr_image'] = _pad_to_max_hw([b['lr_image'] for b in batch])
    collated['query_coords'] = torch.stack([b['query_coords'] for b in batch], dim=0)
    collated['gt_rgb'] = torch.stack([b['gt_rgb'] for b in batch], dim=0)
    collated['cell'] = torch.stack([b['cell'] for b in batch], dim=0)
    collated['scale'] = torch.stack([b['scale'] for b in batch], dim=0)
    collated['full_hr_patch'] = torch.stack([b['full_hr_patch'] for b in batch], dim=0)
    return collated


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
        self.train_loader = DataLoader(
            DIV2KASISRDataset(dcfg['root'], 'train', dcfg['patch_size'], dcfg['query_points'], dcfg['scale_min'], dcfg['scale_max'], dcfg['interpolation_mode']),
            batch_size=self.cfg['train']['batch_size'],
            shuffle=True,
            collate_fn=asisr_collate_fn,
        )
        self.val_loader = DataLoader(
            DIV2KASISRDataset(dcfg['root'], 'val', dcfg['patch_size'], dcfg['query_points'], dcfg['scale_min'], dcfg['scale_max'], dcfg['interpolation_mode']),
            batch_size=1,
            collate_fn=asisr_collate_fn,
        )

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
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

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
