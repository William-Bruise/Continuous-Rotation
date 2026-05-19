import torch.nn as nn
from .encoders import BaseEncoder
from .samplers import LocalFeatureSampler
from .decoders import ImplicitDecoder


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
