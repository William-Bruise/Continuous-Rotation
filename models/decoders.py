import torch
import torch.nn as nn


class ImplicitDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim=256, depth=4, out_dim=3):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(depth - 1):
            layers += [nn.Linear(d, hidden_dim), nn.GELU()]
            d = hidden_dim
        layers += [nn.Linear(d, out_dim)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class GroupImplicitDecoder(nn.Module):
    """Group-aware decoder with group token embedding + pooling + RGB head."""
    def __init__(self, in_dim, token_dim=256, hidden_dim=256, token_depth=3, decode_depth=3, pooling='mean'):
        super().__init__()
        self.pooling = pooling
        tok = []
        d = in_dim
        for _ in range(token_depth - 1):
            tok += [nn.Linear(d, token_dim), nn.GELU()]
            d = token_dim
        tok += [nn.Linear(d, token_dim)]
        self.token_mlp = nn.Sequential(*tok)
        self.attn = nn.Linear(token_dim, 1) if pooling == 'attention' else None
        dec = []
        d = token_dim
        for _ in range(decode_depth - 1):
            dec += [nn.Linear(d, hidden_dim), nn.GELU()]
            d = hidden_dim
        dec += [nn.Linear(d, 3)]
        self.decode_mlp = nn.Sequential(*dec)

    def forward(self, z):
        # z: [B,N,G,D]
        b, n, g, d = z.shape
        h = self.token_mlp(z.reshape(b * n * g, d)).view(b, n, g, -1)
        if self.pooling == 'attention':
            a = torch.softmax(self.attn(h), dim=2)
            pooled = (a * h).sum(dim=2)
        else:
            pooled = h.mean(dim=2)
        return self.decode_mlp(pooled)


class O2ImplicitDecoder(nn.Module):
    """Placeholder for future strict O(2)-equivariant implicit decoder."""
    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError('O2ImplicitDecoder is a planned interface placeholder.')
