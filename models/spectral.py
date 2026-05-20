import math
import torch
import torch.nn as nn


class SpectralAngularMix(nn.Module):
    """Band-limited angular Fourier mixing on discrete rotation bins.

    Mathematical role in this repo:
    - Continuous target: angular dependence over theta in O(2).
    - Current implementation: discrete quadrature over K bins + finite mode truncation up to M.
    - Therefore this is an approximation layer (not a final continuous O(2) analytic operator).
    """
    def __init__(self, channels, K=8, num_modes=3, use_reflection=True):
        super().__init__()
        self.channels = channels
        self.K = K
        self.M = min(num_modes, K // 2)
        self.use_reflection = use_reflection
        self.mode_mix = nn.ModuleList([nn.Linear(channels, channels, bias=False) for _ in range(self.M + 1)])

        k = torch.arange(K).float()
        cos_basis = [torch.ones(K)]
        sin_basis = [torch.zeros(K)]
        for m in range(1, self.M + 1):
            ang = 2 * math.pi * m * k / K
            cos_basis.append(torch.cos(ang))
            sin_basis.append(torch.sin(ang))
        self.register_buffer('cos_basis', torch.stack(cos_basis, dim=0))  # [M+1,K]
        self.register_buffer('sin_basis', torch.stack(sin_basis, dim=0))

    def _project_reconstruct(self, x):
        # x: [B,K,C,H,W]
        # Discrete projection/reconstruction on rotation bins (quadrature-style sum over k).
        b, k, c, h, w = x.shape
        out = 0.0
        for m in range(self.M + 1):
            cm = self.cos_basis[m].view(1, k, 1, 1, 1)
            sm = self.sin_basis[m].view(1, k, 1, 1, 1)
            a = (x * cm).sum(dim=1) / self.K  # [B,C,H,W]
            a = self.mode_mix[m](a.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            out = out + a.unsqueeze(1) * cm
            if m > 0:
                bcoef = (x * sm).sum(dim=1) / self.K
                bcoef = self.mode_mix[m](bcoef.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
                out = out + bcoef.unsqueeze(1) * sm
        return out

    def forward(self, fg):
        # fg: [B,G,C,H,W], G=R*K where R in {1,2}
        b, g, c, h, w = fg.shape
        R = 2 if self.use_reflection else 1  # reflection states
        assert g == R * self.K, f'expected group dim {R*self.K}, got {g}'
        x = fg.view(b, R, self.K, c, h, w)
        if R == 2:
            f0, f1 = x[:, 0], x[:, 1]
            even = 0.5 * (f0 + f1)
            odd = 0.5 * (f0 - f1)
            even_t = self._project_reconstruct(even)
            odd_t = self._project_reconstruct(odd)
            y0 = even_t + odd_t
            y1 = even_t - odd_t
            y = torch.stack([y0, y1], dim=1)
        else:
            y = self._project_reconstruct(x[:, 0]).unsqueeze(1)
        return y.view(b, g, c, h, w)


class SpectralGroupBlock(nn.Module):
    def __init__(self, channels, K=8, num_modes=3, use_reflection=True, residual=True):
        super().__init__()
        self.mix = SpectralAngularMix(channels, K=K, num_modes=num_modes, use_reflection=use_reflection)
        self.norm = nn.GroupNorm(1, channels)
        self.residual = residual

    def forward(self, fg):
        b, g, c, h, w = fg.shape
        y = self.mix(fg)
        y = self.norm(y.view(b * g, c, h, w)).view(b, g, c, h, w)
        return fg + y if self.residual else y


class SpectralImplicitAdapter(nn.Module):
    """Stack spectral group blocks on F_G before sampling."""
    def __init__(self, channels, K=8, use_reflection=True, num_modes=3, num_blocks=2, residual=True):
        super().__init__()
        self.blocks = nn.ModuleList([
            SpectralGroupBlock(channels, K=K, num_modes=num_modes, use_reflection=use_reflection, residual=residual)
            for _ in range(num_blocks)
        ])

    def forward(self, fg):
        for blk in self.blocks:
            fg = blk(fg)
        return fg
