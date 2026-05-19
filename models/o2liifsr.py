import torch.nn as nn
from .encoders import BaseEncoder, EquivariantEncoder
from .samplers import LocalFeatureSampler, GroupLocalFeatureSampler
from .decoders import ImplicitDecoder, GroupImplicitDecoder


class O2LIIFSR(nn.Module):
    def __init__(self, feat_ch=64, n_blocks=8, decoder_hidden=256):
        super().__init__()
        self.encoder = BaseEncoder(feat_ch=feat_ch, n_blocks=n_blocks)
        self.sampler = LocalFeatureSampler(include_cell=True, include_scale=True)
        self.decoder = ImplicitDecoder(in_dim=feat_ch + 2 + 2 + 1, hidden_dim=decoder_hidden, depth=5, out_dim=3)

    def forward(self, lr_image, query_coords, cell=None, scale=None):
        feat = self.encoder(lr_image)
        z = self.sampler(feat, query_coords, cell=cell, scale=scale)
        pred = self.decoder(z.reshape(-1, z.shape[-1])).reshape(z.shape[0], z.shape[1], 3)
        return pred


class GroupO2LIIFSR(nn.Module):
    """Discrete O(2)-approximate group-aware LIIF model (lifting-based)."""
    def __init__(self, feat_ch=64, n_blocks=8, decoder_hidden=256, K=8, pooling='mean', reflect_axis='x', use_group_decoder=True):
        super().__init__()
        self.group_encoder = EquivariantEncoder(feat_ch=feat_ch, n_blocks=n_blocks, K=K, reflect_axis=reflect_axis)
        self.group_sampler = GroupLocalFeatureSampler(include_cell=True, include_scale=True)
        self.use_group_decoder = use_group_decoder
        in_dim = feat_ch + 2 + 2 + 1
        self.group_decoder = GroupImplicitDecoder(in_dim=in_dim, token_dim=decoder_hidden, hidden_dim=decoder_hidden, pooling=pooling)
        self.baseline_decoder = ImplicitDecoder(in_dim=in_dim, hidden_dim=decoder_hidden, depth=5, out_dim=3)

    def forward(self, lr_image, query_coords, cell=None, scale=None):
        f_g = self.group_encoder(lr_image)  # [B,G,C,H,W]
        z = self.group_sampler(f_g, query_coords, cell=cell, scale=scale)  # [B,N,G,D]
        if self.use_group_decoder:
            return self.group_decoder(z)
        z_mean = z.mean(dim=2)
        return self.baseline_decoder(z_mean.reshape(-1, z_mean.shape[-1])).reshape(z_mean.shape[0], z_mean.shape[1], 3)
