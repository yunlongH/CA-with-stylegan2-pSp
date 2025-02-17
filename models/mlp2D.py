import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import List

class EqualizedLinear(nn.Module):
    """
    Equalized Learning-Rate Linear Layer with initialized weight scaling.
    """
    def __init__(self, in_features: int, out_features: int, bias: float = 0.0):
        super().__init__()
        self.weight = EqualizedWeight([out_features, in_features])
        self.bias = nn.Parameter(torch.full((out_features,), bias))

    def forward(self, x: torch.Tensor):
        return F.linear(x, self.weight(), bias=self.bias)

class EqualizedWeight(nn.Module):
    """
    Implements learning-rate equalized weight scaling, based on Progressive GANs,
    with weights initialized to `N(0, 1)` and scaled by `c`.
    """
    def __init__(self, shape: List[int]):
        super().__init__()
        self.c = 1 / math.sqrt(np.prod(shape[1:]))
        self.weight = nn.Parameter(torch.randn(shape))

    def forward(self):
        return self.weight * self.c

class MappingNetwork(nn.Module):
    """
    Standard Mapping Network that maps a latent vector `z` to an intermediate
    latent space `w` using equalized learning-rate linear layers and LeakyReLU.
    """
    def __init__(self, features: int, n_layers: int):
        super().__init__()
        self.net = nn.Sequential(*[self._layer(features) for _ in range(n_layers)])

    def _layer(self, features):
        return nn.Sequential(EqualizedLinear(features, features), nn.LeakyReLU(0.2, inplace=True))

    def forward(self, z: torch.Tensor):
        return self.net(F.normalize(z, dim=1))

class MappingNetwork_c2s(nn.Module):
    """
    Standard Mapping Network that maps a latent vector `z` to an intermediate
    latent space `w` using equalized learning-rate linear layers and LeakyReLU.
    """
    def __init__(self, features: int, n_layers: int):
        super().__init__()
        self.net = nn.Sequential(*[self._layer(features) for _ in range(n_layers)])

    def _layer(self, features):
        return nn.Sequential(EqualizedLinear(features, features), nn.LeakyReLU(0.2, inplace=True))

    def forward(self, z: torch.Tensor):
        return self.net(F.normalize(z, dim=1))

class MappingNetwork_cs(nn.Module):
    """
    Mapping Network with Dual Outputs (`c` and `s`), mapping `z` into two separate spaces
    through parallel networks with equalized learning-rate linear layers and LeakyReLU.
    """
    def __init__(self, features: int, n_layers: int):
        super().__init__()
        self.net_c = nn.Sequential(*[self._layer(features) for _ in range(n_layers)])
        self.net_s = nn.Sequential(*[self._layer(features) for _ in range(n_layers)])

    def _layer(self, features):
        return nn.Sequential(EqualizedLinear(features, features), nn.LeakyReLU(0.2, inplace=True))

    def forward(self, z: torch.Tensor):
        z = F.normalize(z, dim=1)
        return self.net_c(z), self.net_s(z)


class MappingNetwork_cs_individual(nn.Module):
    """
    Mapping Network for Individual Controlled Sparsity
    that generates `c` and `s` for each feature in `z`.
    """
    def __init__(self, features: int, n_layers: int):
        super().__init__()
        self.net_c = nn.Sequential(*[self._layer(features) for _ in range(n_layers)])
        self.net_s = nn.Sequential(*[self._layer(features) for _ in range(n_layers)])

    def _layer(self, features):
        return nn.Sequential(EqualizedLinear(features, features), nn.LeakyReLU(0.2, inplace=True))

    def forward(self, z: torch.Tensor):
        z = F.normalize(z, dim=1)
        c_list, s_list = zip(*[(self.net_c(z[:, i, :]), self.net_s(z[:, i, :])) for i in range(z.size(1))])
        return torch.stack(c_list, dim=1), torch.stack(s_list, dim=1)


class MappingNetwork_cs_pyramid(nn.Module):
    """
    Pyramid Mapping Network, generating levels `c` and `s` mappings for each defined
    level split (coarse, middle, fine).
    """
    def __init__(self, features: int, n_layers: int):
        super().__init__()
        self.coarse_ind, self.middle_ind = 3, 7
        self.net_c = nn.Sequential(*[self._layer(features) for _ in range(n_layers)])
        self.net_s = nn.Sequential(*[self._layer(features) for _ in range(n_layers)])

    def _layer(self, features):
        return nn.Sequential(EqualizedLinear(features, features), nn.LeakyReLU(0.2, inplace=True))

    def forward(self, z: torch.Tensor):
        z = F.normalize(z, dim=1)
        levels = [(0, self.coarse_ind), (self.coarse_ind, self.middle_ind), (self.middle_ind, z.size(1))]
        c_parts, s_parts = zip(*[(self.net_c(z[:, start:end]), self.net_s(z[:, start:end])) for start, end in levels])
        return torch.cat(c_parts, dim=1), torch.cat(s_parts, dim=1)


class MappingNetwork_cs_sparsity(nn.Module):
    """
    Dual-output Mapping Network with Controlled Sparsity in Outputs using `zero_out` method.
    Maps input `z` to outputs `c` and `s` with sparsity control.
    """
    def __init__(self, opts):
        super().__init__()
        self._validate_opts(opts)
        self.opts = opts
        features, n_layers = opts.latent_dim, opts.n_layers_mlp
        self.net_c = nn.Sequential(*[self._layer(features) for _ in range(n_layers)])
        self.net_s = nn.Sequential(*[self._layer(features) for _ in range(n_layers)])
        self.zero_out_type = opts.zero_out_type
        self.zero_out_threshold = opts.zero_out_threshold
        self.norm_type = opts.mlp_norm_type

    def _layer(self, features):
        return nn.Sequential(EqualizedLinear(features, features), nn.LeakyReLU(0.2, inplace=True))

    def _validate_opts(self, opts):
        required_attrs = ["latent_dim", "n_layers_mlp", "zero_out_type", "zero_out_threshold"]
        if not all(hasattr(opts, attr) for attr in required_attrs):
            raise ValueError(f"opts must contain {', '.join(required_attrs)}.")

    def zero_out(self, x):
        threshold_tensor = torch.tensor(0.0, device=x.device)
        if self.zero_out_type == 'hard':
            return torch.where(torch.abs(x) < self.zero_out_threshold, threshold_tensor, x)
        elif self.zero_out_type == 'soft':
            return torch.sign(x) * torch.maximum(torch.abs(x) - self.zero_out_threshold, threshold_tensor)
        else:
            raise ValueError(f"Unknown zero_out_type: {self.zero_out_type}")

    def forward(self, z: torch.Tensor, zero_out_common: bool = False, zero_out_silent: bool = False):
        # Normalize z based on specified norm_type. If norm_type is any other value, do nothing (leave z unchanged)
        if self.norm_type == 'dim1':
            z = F.normalize(z, dim=1)
        elif self.norm_type == 'dim2':
            z = F.normalize(z, dim=2)
        elif self.norm_type == 'dim12':
            z = F.normalize(z, dim=(1, 2))

        c, s = self.net_c(z), self.net_s(z)
        
        if zero_out_common:
            c = self.zero_out(c)
        if zero_out_silent:
            s = self.zero_out(s)
        
        return c, s
