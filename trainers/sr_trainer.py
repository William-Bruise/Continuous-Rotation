import os
import random
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import DIV2KASISRDataset
from models import O2LIIFSR, GroupO2LIIFSR, ContinuousGroupO2LIIFSR
from losses.reconstruction import reconstruction_l1
from losses.equivariance import rotation_consistency_loss, reflection_consistency_loss, continuous_equivariance_loss
from evaluators.sr_evaluator import evaluate_model, save_metrics
from utils.checkpoint import save_checkpoint, load_checkpoint


def _pad_to_max_hw(images):
    max_h = max(x.shape[-2] for x in images)
    max_w = max(x.shape[-1] for x in images)
    return torch.stack([F.pad(x, (0, max_w - x.shape[-1], 0, max_h - x.shape[-2]), mode='replicate') for x in images], dim=0)


def asisr_collate_fn(batch):
    return {
        'lr_image': _pad_to_max_hw([b['lr_image'] for b in batch]),
        'query_coords': torch.stack([b['query_coords'] for b in batch], dim=0),
        'gt_rgb': torch.stack([b['gt_rgb'] for b in batch], dim=0),
        'cell': torch.stack([b['cell'] for b in batch], dim=0),
        'scale': torch.stack([b['scale'] for b in batch], dim=0),
        'full_hr_patch': torch.stack([b['full_hr_patch'] for b in batch], dim=0),
    }


class SRTrainer:
    def __init__(self, cfg, exp_dir, logger):
        self.cfg, self.exp_dir, self.logger = cfg, exp_dir, logger
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self._build_model().to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=cfg['train']['learning_rate'])
        self.start_epoch, self.best_psnr = 0, -1e9
        self._build_data()

    def _build_model(self):
        mcfg = self.cfg['model']
        variant = mcfg.get('variant', 'baseline_liif')
        if variant == 'continuous_so2':
            return ContinuousGroupO2LIIFSR(feat_ch=mcfg['encoder_channels'], n_blocks=mcfg['num_residual_blocks'], decoder_hidden=mcfg['decoder_hidden_dim'], num_fourier_orders=mcfg.get('num_fourier_orders',3), num_orientation_quadrature=mcfg.get('num_orientation_quadrature',8), orientation_quadrature_mode=mcfg.get('orientation_quadrature_mode','uniform'), max_rotation_radians=mcfg.get('max_rotation_radians', 6.283185307179586))
        if variant in ('baseline_liif', 'baseline_liif_consistency'):
            return O2LIIFSR(feat_ch=mcfg['encoder_channels'], n_blocks=mcfg['num_residual_blocks'], decoder_hidden=mcfg['decoder_hidden_dim'])
        use_group_decoder = variant in ('group_encoder_group_decoder', 'group_encoder_group_decoder_consistency')
        return GroupO2LIIFSR(
            feat_ch=mcfg['encoder_channels'], n_blocks=mcfg['num_residual_blocks'], decoder_hidden=mcfg['decoder_hidden_dim'],
            K=mcfg.get('num_rotations', mcfg.get('group_K', 8)), pooling=mcfg.get('group_pooling', 'mean'), reflect_axis=mcfg.get('reflect_axis', 'x'),
            use_group_decoder=use_group_decoder,
            use_reflection=mcfg.get('use_reflection', True),
            num_angular_modes=mcfg.get('num_angular_modes', 3),
            num_spectral_blocks=mcfg.get('num_spectral_blocks', 2),
            spectral_residual=mcfg.get('spectral_residual', True),
            enable_spectral=mcfg.get('enable_spectral', True),
        )

    def _build_data(self):
        dcfg = self.cfg['dataset']
        ds_args = (dcfg['root'], dcfg['patch_size'], dcfg['query_points'], dcfg['scale_min'], dcfg['scale_max'], dcfg['interpolation_mode'])
        self.train_loader = DataLoader(DIV2KASISRDataset(ds_args[0], 'train', *ds_args[1:]), batch_size=self.cfg['train']['batch_size'], shuffle=True, collate_fn=asisr_collate_fn)
        self.val_loader = DataLoader(DIV2KASISRDataset(ds_args[0], 'val', *ds_args[1:]), batch_size=1, collate_fn=asisr_collate_fn)

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
                lr, q, gt = batch['lr_image'].to(self.device), batch['query_coords'].to(self.device), batch['gt_rgb'].to(self.device)
                cell, scale = batch['cell'].to(self.device), batch['scale'].to(self.device)
                pred = self.model(lr, q, cell, scale)
                loss = reconstruction_l1(pred, gt)
                if self.cfg['loss']['enable_rot']:
                    loss = loss + self.cfg['loss']['lambda_rot'] * rotation_consistency_loss(self.model, lr, q, cell, scale, random.choice(self.cfg['eval']['angles']))
                if self.cfg['loss']['enable_ref']:
                    loss = loss + self.cfg['loss']['lambda_ref'] * reflection_consistency_loss(self.model, lr, q, cell, scale, axis=self.cfg['model'].get('reflect_axis', 'x'))

                if self.cfg['loss'].get('use_continuous_eq_loss', False):
                    loss = loss + self.cfg['loss'].get('continuous_eq_loss_weight', 0.05) * continuous_equivariance_loss(
                        self.model, lr, q, cell, scale,
                        max_rotation_radians=self.cfg['model'].get('max_rotation_radians', 6.283185307179586),
                        interp_mode=self.cfg['model'].get('rotation_interp_mode', 'bilinear')
                    )

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
