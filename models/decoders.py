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


class O2ImplicitDecoder(nn.Module):
    """Placeholder for future strict O(2)-equivariant implicit decoder."""
    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError('O2ImplicitDecoder is a planned interface placeholder.')
