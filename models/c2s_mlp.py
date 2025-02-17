import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import List


class EqualizedLinear(nn.Module):
    """
    Equalized Learning-Rate Linear Layer with initialized weight scaling,
    using a unique weight matrix for each style dimension (e.g., 18x512x512).
    """
    def __init__(self, in_features: int, out_features: int, style_dim: int = 18, bias: float = 0.0):
        super().__init__()
        # Create `style_dim` independent `512x512` weight matrices
        self.weight = EqualizedWeight([style_dim, out_features, in_features])
        self.bias = nn.Parameter(torch.full((out_features,), bias))

    def forward(self, x: torch.Tensor):
        # Ensure x has shape [batch_size, style_dim, in_features]
        batch_size, style_dim, _ = x.shape
        assert style_dim == self.weight.weight.shape[0], "Input style dimension does not match weight matrix count."
        
        # Apply each weight matrix at the corresponding style dimension position
        outputs = []
        for i in range(style_dim):
            # Apply the ith weight matrix to the ith vector in the style dimension
            output_i = F.linear(x[:, i, :], self.weight()[i], bias=self.bias)
            outputs.append(output_i)

        # Stack outputs along the style dimension
        return torch.stack(outputs, dim=1)

class EqualizedWeight(nn.Module):
    """
    Implements learning-rate equalized weight scaling with weights initialized as 3D
    (e.g., `style_dim x out_features x in_features`) for each style dimension position.
    """
    def __init__(self, shape: List[int]):
        super().__init__()
        self.c = 1 / math.sqrt(shape[2])  # Scale based on `in_features`
        #self.c = 1 / math.sqrt(np.prod(shape[1:]))  # Scale based on `samples`
        self.weight = nn.Parameter(torch.randn(shape)* 0.01)  # Shape: [style_dim, out_features, in_features]

    def forward(self):
        return self.weight * self.c

class MappingNetwork(nn.Module):
    def __init__(self, features: int, n_layers: int, style_dim: int):
        super().__init__()
        self.net = nn.Sequential(*[self._layer(features, style_dim) for _ in range(n_layers)])

    def _layer(self, features, style_dim):
        return nn.Sequential(EqualizedLinear(features, features, style_dim=style_dim), nn.LeakyReLU(0.2, inplace=True))

    def forward(self, z: torch.Tensor):
        return self.net(F.normalize(z, dim=2))  # Normalize across `in_features`


class MappingNetwork_c2s(nn.Module):

    def __init__(self, style_dim=18, features=512, n_layers=12):
        super().__init__()
 
        self.net_c2s = nn.Sequential(*[self._layer(features, style_dim) for _ in range(n_layers)])

    def _layer(self, features, style_dim):
        # Initialize EqualizedLinear with `style_dim` different 512x512 matrices
        return nn.Sequential(EqualizedLinear(features, features, style_dim=style_dim), nn.LeakyReLU(0.2, inplace=True))

    def forward(self, c: torch.Tensor):
        
        # Pass the normalized input through net_c and net_s
        s = self.net_c2s(c)
        
        return s


class SimpleLinearModel(nn.Module):
    """
    A simple linear model that maps C_t (batch, 18, 512) to S_t (batch, 18, 512)
    using independent linear transformations for each of the 18 style dimensions.
    """
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(512, 512)  # Linear transformation for each latent dimension

    def forward(self, c):
        batch_size, style_dim, feature_dim = c.shape
        assert style_dim == 18 and feature_dim == 512, "Input shape mismatch"

        # Apply the linear transformation to each of the 18 dimensions independently
        s_pred = self.fc(c)  # Shape: (batch, 18, 512)
        return s_pred

class DeepC2SModel(nn.Module):
    """
    A deep non-linear model that maps C_t (batch, 18, 512) to S_t (batch, 18, 512)
    using multiple non-linear transformations for each of the 18 style dimensions.
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(512, 512),  
            nn.LeakyReLU(0.2, inplace=True),  # Use LeakyReLU instead of ReLU
            nn.Linear(512, 512),  
            nn.LeakyReLU(0.2, inplace=True),  # LeakyReLU activation
            nn.Linear(512, 512)   
        )

    def forward(self, c):
        return self.net(c)  # Shape: (batch, 18, 512)

# class StrongerC2SModel(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(512, 1024),
#             nn.ReLU(),
#             nn.Linear(1024, 1024),
#             nn.ReLU(),
#             nn.Linear(1024, 512)  # Output remains same shape as input
#         )

#     def forward(self, c):
#         return self.net(c)  # Shape: (batch, 18, 512)

# import torch
# import torch.nn as nn

class StrongerC2SModel(nn.Module):
    """
    A deeper non-linear model that maps C_t (batch, 18, 512) to S_t (batch, 18, 512)
    with more layers and LeakyReLU activations.
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(512, 1024),  # Expand feature space
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1024),  # Deeper layers
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 512),   # Back to original size
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),    # Additional layer for refining output
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512)     # Output same shape as input
        )

    def forward(self, c):
        return self.net(c)  # Shape: (batch, 18, 512)


# class StrongerC2SModel(nn.Module):
#     """
#     A deeper non-linear model that maps C_t (batch, 18, 512) to S_t (batch, 18, 512)
#     with more layers and LeakyReLU activations.
#     """
#     def __init__(self):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(512, 1024),  # Expand feature space
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Linear(1024, 1024),  # Deeper layers
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Linear(1024, 512),   # Back to original size
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Linear(512, 512),    # Additional layer for refining output
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Linear(512, 512)     # Output same shape as input
#         )

#     def forward(self, c):
#         return self.net(c)  # Shape: (batch, 18, 512)


# import torch
# import torch.nn as nn

class StrongerC2SModel(nn.Module):
    """
    A deeper non-linear model that maps C_t (batch, 18, 512) to S_t (batch, 18, 512)
    with more layers and LeakyReLU activations.
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(512, 1024),  # Expand feature space
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1024),  # Deeper layers
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1024),  # Additional deep layer
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 512),   # Back to original size
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),    # Additional layer for refining output
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),    # Refinement layer
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512)     # Output same shape as input
        )

    def forward(self, c):
        return self.net(c)  # Shape: (batch, 18, 512)

# Example usage:
# model = StrongerC2SModel()
# c_t = torch.randn(4, 18, 512)  # Example input
# s_t = model(c_t)
