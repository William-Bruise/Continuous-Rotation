import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoders import BaseEncoder
from .decoders import ImplicitDecoder


class FourierCoeffEncoder(nn.Module):
    """Continuous-angle coefficient field encoder.

    Theoretical target: latent z(u,theta) over continuous theta.
    Implementation: predict band-limited Fourier coefficients A0, Acos_m, Asin_m.
    """
    def __init__(self, feat_ch=64, n_blocks=8, num_orders=3):
        super().__init__()
        self.num_orders = num_orders
        self.base = BaseEncoder(feat_ch=feat_ch, n_blocks=n_blocks)
        self.h0 = nn.Conv2d(feat_ch, feat_ch, 1)
        self.hcos = nn.Conv2d(feat_ch, feat_ch * num_orders, 1)
        self.hsin = nn.Conv2d(feat_ch, feat_ch * num_orders, 1)

    def forward(self, y):
        f = self.base(y)
        b, c, h, w = f.shape
        a0 = self.h0(f)
        acos = self.hcos(f).view(b, self.num_orders, c, h, w)
        asin = self.hsin(f).view(b, self.num_orders, c, h, w)
        return {'a0': a0, 'acos': acos, 'asin': asin}


def _sample_map(x, coords):
    b, n, _ = coords.shape
    grid = coords.view(b, n, 1, 2)
    return F.grid_sample(x, grid, mode='bilinear', align_corners=False).squeeze(-1).transpose(1, 2)


def sample_orientation_feature(coeffs, coords, theta):
    """Compute z(p,theta)=a0(p)+sum_m a_m(p)cos(m theta)+b_m(p)sin(m theta)."""
    a0 = _sample_map(coeffs['a0'], coords)  # [B,N,C]
    acos = coeffs['acos']
    asin = coeffs['asin']
    b, m, c, h, w = acos.shape
    n = coords.shape[1]
    acos_s = torch.stack([_sample_map(acos[:, i], coords) for i in range(m)], dim=1)  # [B,M,N,C]
    asin_s = torch.stack([_sample_map(asin[:, i], coords) for i in range(m)], dim=1)

    if not torch.is_tensor(theta):
        theta = torch.tensor(theta, device=coords.device, dtype=coords.dtype).view(1, 1, 1).repeat(b, n, 1)
    elif theta.dim() == 0:
        theta = theta.view(1, 1, 1).repeat(b, n, 1)
    elif theta.dim() == 1:
        theta = theta.view(b, 1, 1).repeat(1, n, 1)
    theta = theta.to(coords.device, coords.dtype)

    out = a0
    for i in range(m):
        order = i + 1
        ct = torch.cos(order * theta).unsqueeze(-1)
        st = torch.sin(order * theta).unsqueeze(-1)
        out = out + acos_s[:, i] * ct + asin_s[:, i] * st
    return out


class OrientationQuadratureAggregator(nn.Module):
    def __init__(self, num_quadrature=8, mode='uniform', max_rotation_radians=2 * math.pi):
        super().__init__()
        self.num_quadrature = num_quadrature
        self.mode = mode
        self.max_rotation_radians = max_rotation_radians

    def sample_thetas(self, b, n, device, dtype):
        q = self.num_quadrature
        if self.mode == 'uniform':
            base = torch.linspace(0, self.max_rotation_radians, q + 1, device=device, dtype=dtype)[:-1]
            return base.view(1, 1, q, 1).repeat(b, n, 1, 1)
        return torch.rand(b, n, q, 1, device=device, dtype=dtype) * self.max_rotation_radians

    def forward(self, coeffs, coords):
        b, n, _ = coords.shape
        thetas = self.sample_thetas(b, n, coords.device, coords.dtype)
        zs = []
        for i in range(thetas.shape[2]):
            zs.append(sample_orientation_feature(coeffs, coords, thetas[:, :, i]))
        z = torch.stack(zs, dim=2)  # [B,N,Q,C]
        return z.mean(dim=2)


class ContinuousGroupO2LIIFSR(nn.Module):
    """Continuous-angle formulation with numerical approximations.

    Represents latent orientation response by Fourier coefficient fields and queries z(p,theta)
    continuously; quadrature nodes are only numerical integration samples.
    """
    def __init__(self, feat_ch=64, n_blocks=8, decoder_hidden=256, num_fourier_orders=3,
                 num_orientation_quadrature=8, orientation_quadrature_mode='uniform', max_rotation_radians=2 * math.pi):
        super().__init__()
        self.coeff_encoder = FourierCoeffEncoder(feat_ch=feat_ch, n_blocks=n_blocks, num_orders=num_fourier_orders)
        self.agg = OrientationQuadratureAggregator(num_quadrature=num_orientation_quadrature, mode=orientation_quadrature_mode,
                                                   max_rotation_radians=max_rotation_radians)
        self.decoder = ImplicitDecoder(in_dim=feat_ch + 2 + 2 + 1, hidden_dim=decoder_hidden, depth=5, out_dim=3)

    def forward(self, lr_image, query_coords, cell=None, scale=None):
        coeffs = self.coeff_encoder(lr_image)
        q = self.agg(coeffs, query_coords)
        parts = [q, query_coords]
        if cell is not None:
            parts.append(cell)
        if scale is not None:
            parts.append(scale.unsqueeze(1).repeat(1, query_coords.shape[1], 1))
        z = torch.cat(parts, dim=-1)
        return self.decoder(z.reshape(-1, z.shape[-1])).reshape(z.shape[0], z.shape[1], 3)
