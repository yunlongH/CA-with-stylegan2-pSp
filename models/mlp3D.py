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
        self.weight = nn.Parameter(torch.randn(shape))  # Shape: [style_dim, out_features, in_features]

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



class LeakyReLUTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # LeakyReLU activation with slope 0.2 and inplace set to True
        self.activation_fn = nn.LeakyReLU(0.2, inplace=True)
    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        # Self-attention block
        src2 = self.self_attn(
            src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask, is_causal=is_causal
        )[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # Feedforward block with LeakyReLU
        src2 = self.linear2(self.dropout(self.activation_fn(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class MappingNetwork_cs_independent_spatial(nn.Module):
    """
    Combines Spatial-Aware Encoding with MLP-based MappingNetwork_cs_independent.
    """
    def __init__(self, opts):
        super().__init__()
        self._validate_opts(opts)
        self.opts = opts
        self.style_dim = opts.style_dim  # Set style_dim from opts
        features, n_layers = opts.latent_dim, opts.n_layers_mlp

        # Spatial-Aware Encoder
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # Add channel dimension
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1)   # Reduce back to single channel
        )

        # Inter-Row Attention
        self.row_attention = nn.MultiheadAttention(embed_dim=features, num_heads=4)

        # MLP Networks for C and S decomposition
        self.net_c = nn.Sequential(*[self._layer(features, self.style_dim) for _ in range(n_layers)])
        self.net_s = nn.Sequential(*[self._layer(features, self.style_dim) for _ in range(n_layers)])

        self.zero_out_type = opts.zero_out_type
        self.zero_out_threshold = opts.zero_out_threshold
        self.norm_type = opts.mlp_norm_type
        self.spatial_encoding = opts.spatial_encoding

    def _layer(self, features, style_dim):
        # Initialize EqualizedLinear with style_dim different 512x512 matrices
        return nn.Sequential(EqualizedLinear(features, features, style_dim=style_dim), nn.LeakyReLU(0.2, inplace=True))

    def _validate_opts(self, opts):
        required_attrs = ["latent_dim", "n_layers_mlp", "style_dim", "zero_out_type", "zero_out_threshold"]
        if not all(hasattr(opts, attr) for attr in required_attrs):
            raise ValueError(f"opts must contain {', '.join(required_attrs)}.")

    def zero_out(self, x):
        threshold_tensor = torch.zeros_like(x)
        if self.zero_out_type == 'hard':
            return torch.where(torch.abs(x) < self.zero_out_threshold, threshold_tensor, x)
        elif self.zero_out_type == 'soft':
            return torch.sign(x) * torch.maximum(torch.abs(x) - self.zero_out_threshold, threshold_tensor)
        else:
            raise ValueError(f"Unknown zero_out_type: {self.zero_out_type}")
        
    def _apply_spatial_aware(self, z: torch.Tensor):
        """Apply spatial-aware preprocessing."""
        z = z.unsqueeze(1)  # Add a channel dimension: [batch_size, 1, 18, 512]
        z = self.spatial_encoder(z)  # Spatial processing
        return z.squeeze(1)  # Remove channel dimension: [batch_size, 18, 512]
        
    def _apply_row_attention(self, z: torch.Tensor):
        """Apply inter-row attention preprocessing."""
        z = z.permute(1, 0, 2)  # Shape: [18, batch_size, 512] (required by MultiheadAttention)
        z, _ = self.row_attention(z, z, z)  # Apply self-attention
        return z.permute(1, 0, 2)  # Shape: [batch_size, 18, 512]

    def forward(self, z: torch.Tensor, zero_out_common: bool = False, zero_out_silent: bool = False):
        # Apply preprocessing based on the selected mode
        if self.spatial_encoding == "spatial-aware":
            z = self._apply_spatial_aware(z)
        elif self.spatial_encoding == "inter-row-attention":
            z = self._apply_row_attention(z)
        elif self.spatial_encoding == "none":
            pass  # No preprocessing
        else:
            raise ValueError(f"Unknown spatial_encoding type: {self.spatial_encoding}")

        # Normalize z based on specified norm_type
        if self.norm_type == 'dim1':
            z = F.normalize(z, dim=1)
        elif self.norm_type == 'dim2':
            z = F.normalize(z, dim=2)
        elif self.norm_type == 'dim12':
            z = F.normalize(z, dim=(1, 2))
        
        # Pass through MLP networks for decomposition
        c, s = self.net_c(z), self.net_s(z)

        # Optionally zero out parts of the outputs
        if zero_out_common:
            c = self.zero_out(c)
        if zero_out_silent:
            s = self.zero_out(s)
        
        return c, s


class ResidualBlock(nn.Module):
    """
    A single Residual Block for Mapping Network with skip connection.
    """
    def __init__(self, features, style_dim):
        super().__init__()
        self.layer1 = EqualizedLinear(features, features, style_dim=style_dim)
        self.activation1 = nn.LeakyReLU(0.2, inplace=True)
        self.layer2 = EqualizedLinear(features, features, style_dim=style_dim)
        self.activation2 = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch_size, style_dim, features].
        Returns:
            Output tensor with skip connection applied.
        """
        residual = x  # Save the input for the skip connection
        out = self.activation1(self.layer1(x))
        out = self.activation2(self.layer2(out))
        return out + residual  # Add the skip connection

class SampleResidualBlock(nn.Module):
    """
    Lightweight Residual Block with a single transformation and skip connection.
    """
    def __init__(self, features, style_dim):
        super().__init__()
        self.layer1 = EqualizedLinear(features, features, style_dim=style_dim)
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch_size, style_dim, features].
        Returns:
            Output tensor with skip connection applied.
        """
        return x + self.activation(self.layer1(x))  # Single transformation with skip connection


class MappingNetwork_cs_independent_ResNet(nn.Module):
    """
    ResNet-style Mapping Network with the same total number of layers as the original MLP.
    """
    def __init__(self, opts):
        super().__init__()
        self._validate_opts(opts)
        self.opts = opts
        self.style_dim = opts.style_dim
        features, n_blocks = opts.latent_dim, opts.n_layers_mlp 

        if opts.resblock_type == 'standard':  # 2 layers per block
            n_blocks = n_blocks // 2
            self.net_c = nn.Sequential(*[ResidualBlock(features, self.style_dim) for _ in range(n_blocks)])
            self.net_s = nn.Sequential(*[ResidualBlock(features, self.style_dim) for _ in range(n_blocks)])
        elif opts.resblock_type == 'simple':  # 1 layer per block
            self.net_c = nn.Sequential(*[SampleResidualBlock(features, self.style_dim) for _ in range(n_blocks)])
            self.net_s = nn.Sequential(*[SampleResidualBlock(features, self.style_dim) for _ in range(n_blocks)])
        else:
            raise ValueError(f"Unknown resblock_type: {opts.resblock_type}")

        self.zero_out_type = opts.zero_out_type
        self.zero_out_threshold = opts.zero_out_threshold
        self.norm_type = opts.mlp_norm_type

    def _validate_opts(self, opts):
        required_attrs = ["latent_dim", "n_layers_mlp", "style_dim", "zero_out_type", "zero_out_threshold"]
        if not all(hasattr(opts, attr) for attr in required_attrs):
            raise ValueError(f"opts must contain {', '.join(required_attrs)}.")

    def zero_out(self, x):
        threshold_tensor = torch.zeros_like(x)
        if self.zero_out_type == 'hard':
            return torch.where(torch.abs(x) < self.zero_out_threshold, threshold_tensor, x)
        elif self.zero_out_type == 'soft':
            return torch.sign(x) * torch.maximum(torch.abs(x) - self.zero_out_threshold, threshold_tensor)
        else:
            raise ValueError(f"Unknown zero_out_type: {self.zero_out_type}")

    def forward(self, z: torch.Tensor, zero_out_common: bool = False, zero_out_silent: bool = False):
        """
        Args:
            z: Input tensor of shape [batch_size, style_dim, latent_dim].
        Returns:
            c: Tensor of shape [batch_size, style_dim, latent_dim].
            s: Tensor of shape [batch_size, style_dim, latent_dim].
        """
        # Normalize z based on specified norm_type
        if self.norm_type == 'dim1':
            z = F.normalize(z, dim=1)
        elif self.norm_type == 'dim2':
            z = F.normalize(z, dim=2)
        elif self.norm_type == 'dim12':
            z = F.normalize(z, dim=(1, 2))

        # Pass the normalized input through net_c and net_s
        c = self.net_c(z)
        s = self.net_s(z)

        # Apply zero-out sparsity if enabled
        if zero_out_common:
            c = self.zero_out(c)
        if zero_out_silent:
            s = self.zero_out(s)

        return c, s


class CNNMappingNetwork(nn.Module):
    """
    CNN-based Mapping Network for decomposing W+ latent space.
    Processes `net_c` and `net_s` independently with an encoder-decoder structure.
    """
    def __init__(self, opts):
        super().__init__()
        self.opts = opts
        self.zero_out_type = opts.zero_out_type
        self.zero_out_threshold = opts.zero_out_threshold

        # Define independent CNN encoder-decoder architectures for C and S
        self.net_c = self._build_cnn()
        self.net_s = self._build_cnn()

    def _conv_block(self, in_channels, out_channels):
        """
        Convolutional block: Conv2d -> LeakyReLU -> Conv2d -> LeakyReLU
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def _build_cnn(self):
        """
        Build an encoder-decoder CNN architecture for either net_c or net_s.
        """
        # Encoder
        enc1 = self._conv_block(1, 32)  # Input: [B, 1, 18, 512] -> [B, 32, 18, 512]
        enc2 = self._conv_block(32, 64)  # Downsample: [B, 64, 9, 256]
        enc3 = self._conv_block(64, 128)  # Downsample: [B, 128, 4, 128]

        # Bottleneck
        bottleneck = self._conv_block(128, 128)  # [B, 128, 4, 128]

        # Decoder
        dec3 = self._conv_block(128, 64)  # Upsample: [B, 64, 9, 256]
        dec2 = self._conv_block(64, 32)  # Upsample: [B, 32, 18, 512]

        # Final layer
        final = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=3, padding=1),  # Reduce to [B, 1, 18, 512]
            nn.LeakyReLU(0.2, inplace=True),  # Final activation for consistency
        )

        return nn.ModuleDict({
            "enc1": enc1,
            "enc2": enc2,
            "enc3": enc3,
            "bottleneck": bottleneck,
            "dec3": dec3,
            "dec2": dec2,
            "final": final,
        })

    def zero_out(self, x):
        """
        Apply sparsity to outputs.
        """
        threshold_tensor = torch.zeros_like(x)
        if self.zero_out_type == 'hard':
            return torch.where(torch.abs(x) < self.zero_out_threshold, threshold_tensor, x)
        elif self.zero_out_type == 'soft':
            return torch.sign(x) * torch.maximum(torch.abs(x) - self.zero_out_threshold, threshold_tensor)
        else:
            return x

    def forward_single(self, z, net):
        """
        Forward pass for a single CNN (either net_c or net_s).
        """
        # Add channel dimension
        z = z.unsqueeze(1)  # [B, 1, 18, 512]

        # Encoder
        e1 = net["enc1"](z)  # [B, 32, 18, 512]
        e2 = net["enc2"](F.max_pool2d(e1, kernel_size=2))  # [B, 64, 9, 256]
        e3 = net["enc3"](F.max_pool2d(e2, kernel_size=2))  # [B, 128, 4, 128]

        # Bottleneck
        bottleneck = net["bottleneck"](e3)  # [B, 128, 4, 128]

        # Decoder
        d3 = net["dec3"](F.interpolate(bottleneck, scale_factor=2, mode='bilinear', align_corners=False))  # [B, 64, 9, 256]
        d2 = net["dec2"](F.interpolate(d3, scale_factor=2, mode='bilinear', align_corners=False))  # [B, 32, 18, 512]

        # Final layer
        out = net["final"](d2)  # [B, 1, 18, 512]
        return out

    def forward(self, z, zero_out_common=False, zero_out_silent=False):
        # Process through independent CNNs
        c = self.forward_single(z, self.net_c)
        s = self.forward_single(z, self.net_s)

        # Optionally apply sparsity
        if zero_out_common:
            c = self.zero_out(c)
        if zero_out_silent:
            s = self.zero_out(s)

        return c, s


class MappingNetwork_cs_noact_lastlayer(nn.Module):
    """
    Dual-output Mapping Network with Controlled Sparsity in Outputs using `zero_out` method,
    with unique weights per style dimension (style_dim x 512 x 512).
    """
    def __init__(self, opts):
        super().__init__()
        self._validate_opts(opts)
        self.opts = opts
        self.style_dim = opts.style_dim  # Set style_dim from opts
        features, n_layers = opts.latent_dim, opts.n_layers_mlp
        self.net_c = nn.Sequential(*[self._layer(features, self.style_dim, is_last=(i == n_layers - 1)) for i in range(n_layers)])
        self.net_s = nn.Sequential(*[self._layer(features, self.style_dim, is_last=(i == n_layers - 1)) for i in range(n_layers)])
        self.zero_out_type = opts.zero_out_type
        self.zero_out_threshold = opts.zero_out_threshold
        self.norm_type = opts.mlp_norm_type

    def _layer(self, features, style_dim, is_last):
        # Initialize EqualizedLinear with `style_dim` different 512x512 matrices
        layers = [EqualizedLinear(features, features, style_dim=style_dim)]
        if not is_last:
            layers.append(nn.LeakyReLU(0.2, inplace=True))  # Add activation only if not the last layer
        return nn.Sequential(*layers)

    def _validate_opts(self, opts):
        required_attrs = ["latent_dim", "n_layers_mlp", "style_dim", "zero_out_type", "zero_out_threshold"]
        if not all(hasattr(opts, attr) for attr in required_attrs):
            raise ValueError(f"opts must contain {', '.join(required_attrs)}.")

    def zero_out(self, x):
        threshold_tensor = torch.zeros_like(x)
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
        
        # Pass the normalized input through net_c and net_s
        c, s = self.net_c(z), self.net_s(z)
        
        if zero_out_common:
            c = self.zero_out(c)
        if zero_out_silent:
            s = self.zero_out(s)
        
        return c, s

class MappingNetwork_cs_independent(nn.Module):
    """
    Dual-output Mapping Network with Controlled Sparsity in Outputs using `zero_out` method,
    with unique weights per style dimension (style_dim x 512 x 512).
    """
    def __init__(self, opts):
        super().__init__()
        self._validate_opts(opts)
        self.opts = opts
        self.style_dim = opts.style_dim  # Set style_dim from opts
        features, n_layers = opts.latent_dim, opts.n_layers_mlp
        self.net_c = nn.Sequential(*[self._layer(features, self.style_dim) for _ in range(n_layers)])
        self.net_s = nn.Sequential(*[self._layer(features, self.style_dim) for _ in range(n_layers)])
        self.zero_out_type = opts.zero_out_type
        self.zero_out_threshold = opts.zero_out_threshold
        self.norm_type = opts.mlp_norm_type

    def _layer(self, features, style_dim):
        # Initialize EqualizedLinear with `style_dim` different 512x512 matrices
        return nn.Sequential(EqualizedLinear(features, features, style_dim=style_dim), nn.LeakyReLU(0.2, inplace=True))

    def _validate_opts(self, opts):
        required_attrs = ["latent_dim", "n_layers_mlp", "style_dim", "zero_out_type", "zero_out_threshold"]
        if not all(hasattr(opts, attr) for attr in required_attrs):
            raise ValueError(f"opts must contain {', '.join(required_attrs)}.")

    def zero_out(self, x):
        threshold_tensor = torch.zeros_like(x)
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
        
        # Pass the normalized input through net_c and net_s
        c, s = self.net_c(z), self.net_s(z)
        
        if zero_out_common:
            c = self.zero_out(c)
        if zero_out_silent:
            s = self.zero_out(s)
        
        return c, s


class MappingNetwork_c2s(nn.Module):

    def __init__(self, opts):
        super().__init__()
 
        style_dim = opts.style_dim 
        features = opts.latent_dim
        n_layers = opts.n_c2s_layers
        
        self.net_c2s = nn.Sequential(*[self._layer(features, style_dim) for _ in range(n_layers)])

    def _layer(self, features, style_dim):
        # Initialize EqualizedLinear with `style_dim` different 512x512 matrices
        return nn.Sequential(EqualizedLinear(features, features, style_dim=style_dim), nn.LeakyReLU(0.2, inplace=True))

    def forward(self, c: torch.Tensor):

        # Pass the normalized input through net_c and net_s
        s = self.net_c2s(c)
        
        return s

class MappingNetwork_cs_hierarchical(nn.Module):
    """
    Dual-output Mapping Network with Controlled Sparsity in Outputs using `zero_out` method,
    with unique weights per style dimension (style_dim x 512 x 512).
    """
    def __init__(self, opts):
        super().__init__()
        self._validate_opts(opts)
        self.opts = opts
        self.style_dim = opts.style_dim  # Set style_dim from opts
        self.zero_out_type = opts.zero_out_type
        self.zero_out_threshold = opts.zero_out_threshold
        self.norm_type = opts.mlp_norm_type
        # Define group dimensions
        self.low_dims = 3   # Number of low-level style dimensions (0-2)
        self.mid_dims = 4   # Number of mid-level style dimensions (3-6)
        self.high_dims = self.style_dim - self.low_dims - self.mid_dims # Number of high-level style dimensions (7-17)
        features, n_layers = opts.latent_dim, opts.n_layers_mlp

        # Define independent MLPs for C and S components
        self.net_low_c = nn.Sequential(*[self._layer(features, self.low_dims) for _ in range(n_layers)])
        self.net_mid_c = nn.Sequential(*[self._layer(features, self.mid_dims) for _ in range(n_layers)])
        self.net_high_c = nn.Sequential(*[self._layer(features, self.high_dims) for _ in range(n_layers)])

        self.net_low_s = nn.Sequential(*[self._layer(features, self.low_dims) for _ in range(n_layers)])
        self.net_mid_s = nn.Sequential(*[self._layer(features, self.mid_dims) for _ in range(n_layers)])
        self.net_high_s = nn.Sequential(*[self._layer(features, self.high_dims) for _ in range(n_layers)])

    def _layer(self, features, style_dim):
        # Initialize EqualizedLinear with `style_dim` different 512x512 matrices
        return nn.Sequential(EqualizedLinear(features, features, style_dim=style_dim), nn.LeakyReLU(0.2, inplace=True))

    def _validate_opts(self, opts):
        required_attrs = ["latent_dim", "n_layers_mlp", "style_dim", "zero_out_type", "zero_out_threshold"]
        if not all(hasattr(opts, attr) for attr in required_attrs):
            raise ValueError(f"opts must contain {', '.join(required_attrs)}.")

    def zero_out(self, x):
        threshold_tensor = torch.zeros_like(x)
        if self.zero_out_type == 'hard':
            return torch.where(torch.abs(x) < self.zero_out_threshold, threshold_tensor, x)
        elif self.zero_out_type == 'soft':
            return torch.sign(x) * torch.maximum(torch.abs(x) - self.zero_out_threshold, threshold_tensor)
        else:
            raise ValueError(f"Unknown zero_out_type: {self.zero_out_type}")

    def forward(self, z: torch.Tensor, zero_out_common: bool = False, zero_out_silent: bool = False):
        # Validate input shape
        assert z.shape[1] == self.style_dim, f"Input style dimension {z.shape[1]} does not match expected {self.style_dim}"

        # Normalize z based on specified norm_type. If norm_type is any other value, do nothing (leave z unchanged)
        if self.norm_type == 'dim1':
            z = F.normalize(z, dim=1)
        elif self.norm_type == 'dim2':
            z = F.normalize(z, dim=2)
        elif self.norm_type == 'dim12':
            z = F.normalize(z, dim=(1, 2))
        elif self.norm_type == 'none':
            pass  # No normalization
        
        # Split input into low, mid, and high levels
        z_low = z[:, :self.low_dims, :]  # dimensions 0-2
        z_mid = z[:, self.low_dims:self.low_dims + self.mid_dims, :]  # dimensions 3-6
        z_high = z[:, self.low_dims + self.mid_dims:, :]  # dimensions 7-18
        # print(f"z_low shape: {z_low.shape}, expected: {self.low_dims}")
        # print(f"z_mid shape: {z_mid.shape}, expected: {self.mid_dims}")
        # print(f"z_high shape: {z_high.shape}, expected: {self.high_dims}")

        # Process low, mid, and high levels independently for C and S
        c_low = self.net_low_c(z_low)
        c_mid = self.net_mid_c(z_mid)
        c_high = self.net_high_c(z_high)

        s_low = self.net_low_s(z_low)
        s_mid = self.net_mid_s(z_mid)
        s_high = self.net_high_s(z_high)

        # Concatenate results for final C and S
        c = torch.cat([c_low, c_mid, c_high], dim=1)
        s = torch.cat([s_low, s_mid, s_high], dim=1)

        if zero_out_common:
            c = self.zero_out(c)
        if zero_out_silent:
            s = self.zero_out(s)
        
        return c, s


class MappingNetwork_cs_SumNet(nn.Module):
    """
    Dual-output Mapping Network with Controlled Sparsity in Outputs using `zero_out` method,
    with unique weights per style dimension (style_dim x 512 x 512).
    """
    def __init__(self, opts):
        super().__init__()
        self._validate_opts(opts)
        self.opts = opts
        self.style_dim = opts.style_dim  # Set style_dim from opts
        features, n_layers = opts.latent_dim, opts.n_layers_mlp
        self.net_c = nn.Sequential(*[self._layer(features, self.style_dim) for _ in range(n_layers)])
        self.net_s = nn.Sequential(*[self._layer(features, self.style_dim) for _ in range(n_layers)])
        self.zero_out_type = opts.zero_out_type
        self.zero_out_threshold = opts.zero_out_threshold
        self.norm_type = opts.mlp_norm_type

        # SumNetwork layers for combining C and S

        self.combined = nn.Sequential(nn.Linear(features * 2, features * 2),
                                      nn.LeakyReLU(negative_slope=0.2),
                                      nn.Linear(features * 2, features),
                                      nn.LeakyReLU(negative_slope=0.2))

    def _layer(self, features, style_dim):
        # Initialize EqualizedLinear with `style_dim` different 512x512 matrices
        return nn.Sequential(EqualizedLinear(features, features, style_dim=style_dim), nn.LeakyReLU(0.2, inplace=True))

    def _validate_opts(self, opts):
        required_attrs = ["latent_dim", "n_layers_mlp", "style_dim", "zero_out_type", "zero_out_threshold"]
        if not all(hasattr(opts, attr) for attr in required_attrs):
            raise ValueError(f"opts must contain {', '.join(required_attrs)}.")

    def zero_out(self, x):
        threshold_tensor = torch.zeros_like(x)
        if self.zero_out_type == 'hard':
            return torch.where(torch.abs(x) < self.zero_out_threshold, threshold_tensor, x)
        elif self.zero_out_type == 'soft':
            return torch.sign(x) * torch.maximum(torch.abs(x) - self.zero_out_threshold, threshold_tensor)
        else:
            raise ValueError(f"Unknown zero_out_type: {self.zero_out_type}")

    def forward(self, z: torch.Tensor, zero_out_common: bool = False, zero_out_silent: bool = False, zero_sbg: bool = False):
        # Normalize z based on specified norm_type. If norm_type is any other value, do nothing (leave z unchanged)
        if self.norm_type == 'dim1':
            z = F.normalize(z, dim=1)
        elif self.norm_type == 'dim2':
            z = F.normalize(z, dim=2)
        elif self.norm_type == 'dim12':
            z = F.normalize(z, dim=(1, 2))
        
        if zero_out_common:
            c = self.zero_out(c)
        if zero_out_silent:
            s = self.zero_out(s)

        # Pass the normalized input through net_c and net_s
        c, s = self.net_c(z), self.net_s(z)
        # Combine c and s using the combined layer
        if zero_sbg:
            combined_input = torch.cat([c, torch.zeros_like(s).to(s.device)], dim=-1)  # Concatenate along the feature dimension
        else:
            combined_input = torch.cat([c, s], dim=-1)
        combined_output = self.combined(combined_input)

        return combined_output, c, s

class MappingNetwork_cs_sparsity(nn.Module):
    """
    Dual-output Mapping Network with Controlled Sparsity in Outputs using `zero_out` method,
    with unique weights per style dimension (style_dim x 512 x 512).
    """
    def __init__(self, opts):
        super().__init__()
        self._validate_opts(opts)
        self.opts = opts
        self.style_dim = opts.style_dim  # Set style_dim from opts
        features, n_layers = opts.latent_dim, opts.n_layers_mlp
        self.net_c = nn.Sequential(*[self._layer(features, self.style_dim) for _ in range(n_layers)])
        self.net_s = nn.Sequential(*[self._layer(features, self.style_dim) for _ in range(n_layers)])
        self.zero_out_type = opts.zero_out_type
        self.zero_out_threshold = opts.zero_out_threshold
        self.norm_type = opts.mlp_norm_type

    def _layer(self, features, style_dim):
        # Initialize EqualizedLinear with `style_dim` different 512x512 matrices
        return nn.Sequential(EqualizedLinear(features, features, style_dim=style_dim), nn.LeakyReLU(0.2, inplace=True))

    def _validate_opts(self, opts):
        required_attrs = ["latent_dim", "n_layers_mlp", "style_dim", "zero_out_type", "zero_out_threshold"]
        if not all(hasattr(opts, attr) for attr in required_attrs):
            raise ValueError(f"opts must contain {', '.join(required_attrs)}.")

    def zero_out(self, x):
        threshold_tensor = torch.zeros_like(x)
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
        
        # Pass the normalized input through net_c and net_s
        c, s = self.net_c(z), self.net_s(z)
        
        if zero_out_common:
            c = self.zero_out(c)
        if zero_out_silent:
            s = self.zero_out(s)
        
        return c, s
    
class MappingNetwork_cs_shared(nn.Module):
    """
    Dual-output Mapping Network with shared layers for both C and S.
    The shared layers process input Z, and separate heads generate C and S outputs.
    """
    def __init__(self, opts):
        super().__init__()
        self._validate_opts(opts)
        self.opts = opts
        self.style_dim = opts.style_dim  # Set style_dim from opts
        features, n_layers = opts.latent_dim, opts.n_layers_mlp

        # Shared backbone for processing input
        self.shared_net = nn.Sequential(*[self._layer(features, self.style_dim) for _ in range(n_layers)])

        # Independent heads for C and S outputs
        self.head_c = nn.Sequential(EqualizedLinear(features, features, style_dim=self.style_dim), nn.LeakyReLU(0.2, inplace=True))
        self.head_s = nn.Sequential(EqualizedLinear(features, features, style_dim=self.style_dim), nn.LeakyReLU(0.2, inplace=True))

        self.zero_out_type = opts.zero_out_type
        self.zero_out_threshold = opts.zero_out_threshold
        self.norm_type = opts.mlp_norm_type

    def _layer(self, features, style_dim):
        # Shared layers use EqualizedLinear
        return nn.Sequential(EqualizedLinear(features, features, style_dim=style_dim), nn.LeakyReLU(0.2, inplace=True))

    def _validate_opts(self, opts):
        required_attrs = ["latent_dim", "n_layers_mlp", "style_dim", "zero_out_type", "zero_out_threshold"]
        if not all(hasattr(opts, attr) for attr in required_attrs):
            raise ValueError(f"opts must contain {', '.join(required_attrs)}.")

    def zero_out(self, x):
        threshold_tensor = torch.zeros_like(x)
        if self.zero_out_type == 'hard':
            return torch.where(torch.abs(x) < self.zero_out_threshold, threshold_tensor, x)
        elif self.zero_out_type == 'soft':
            return torch.sign(x) * torch.maximum(torch.abs(x) - self.zero_out_threshold, threshold_tensor)
        else:
            raise ValueError(f"Unknown zero_out_type: {self.zero_out_type}")

    def forward(self, z: torch.Tensor, zero_out_common: bool = False, zero_out_silent: bool = False):
        # Normalize z based on specified norm_type. If norm_type is any other value, leave z unchanged.
        if self.norm_type == 'dim1':
            z = F.normalize(z, dim=1)
        elif self.norm_type == 'dim2':
            z = F.normalize(z, dim=2)
        elif self.norm_type == 'dim12':
            z = F.normalize(z, dim=(1, 2))

        # Pass through the shared backbone
        shared_features = self.shared_net(z)

        # Generate C and S using separate heads
        c = self.head_c(shared_features)
        s = self.head_s(shared_features)

        # Optionally zero out parts of the outputs
        if zero_out_common:
            c = self.zero_out(c)
        if zero_out_silent:
            s = self.zero_out(s)

        return c, s





class FactorizationNetwork_cs(nn.Module):
    """
    Factorization Network with Controlled Sparsity in Outputs, similar to MappingNetwork_cs_independent.
    It has separate branches for common (C) and specific (S) factors.
    """
    def __init__(self, opts):
        super().__init__()
        self._validate_opts(opts)
        self.opts = opts
        self.style_dim = opts.style_dim  # Set style_dim from opts
        features, n_shared_layers, n_branch_layers = opts.latent_dim, opts.n_shared_layers, opts.n_branch_layers

        self.shared_net = nn.Sequential(*[self._layer(features, self.style_dim) for _ in range(n_shared_layers)])
        
        self.net_c = nn.Sequential(*[self._layer(features, self.style_dim) for _ in range(n_branch_layers)])
        self.net_s = nn.Sequential(*[self._layer(features, self.style_dim) for _ in range(n_branch_layers)])

        self.zero_out_type = opts.zero_out_type
        self.zero_out_threshold = opts.zero_out_threshold
        self.norm_type = opts.mlp_norm_type

    def _layer(self, features, style_dim):
        # Initialize EqualizedLinear with `style_dim` different 512x512 matrices
        return nn.Sequential(EqualizedLinear(features, features, style_dim=style_dim), nn.LeakyReLU(0.2, inplace=True))

    def _validate_opts(self, opts):
        required_attrs = ["latent_dim", "n_shared_layers", "n_branch_layers", "style_dim", "zero_out_type", "zero_out_threshold"]
        if not all(hasattr(opts, attr) for attr in required_attrs):
            raise ValueError(f"opts must contain {', '.join(required_attrs)}.")
        
    def zero_out(self, x):
        threshold_tensor = torch.zeros_like(x)
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
        
        # Pass through shared backbone
        shared_features = self.shared_net(z)   

        # Pass through branches for C and S
        c = self.net_c(shared_features)
        s = self.net_s(shared_features)
        
        if zero_out_common:
            c = self.zero_out(c)
        if zero_out_silent:
            s = self.zero_out(s)
        
        return c, s


# class MappingNetwork3D_cs(nn.Module):
#     def __init__(self, features: int, n_layers: int, style_dim: int):
#         super().__init__()
#         self.net_c = nn.Sequential(*[self._layer(features, style_dim) for _ in range(n_layers)])
#         self.net_s = nn.Sequential(*[self._layer(features, style_dim) for _ in range(n_layers)])

#     def _layer(self, features, style_dim):
#         return nn.Sequential(EqualizedLinear(features, features, style_dim=style_dim), nn.LeakyReLU(0.2, inplace=True))

#     def forward(self, z: torch.Tensor):
#         z = F.normalize(z, dim=2)  # Normalize across `in_features`
#         return self.net_c(z), self.net_s(z)

# class MappingNetwork3D_cs_individual(nn.Module):
#     def __init__(self, features: int, n_layers: int, style_dim: int):
#         super().__init__()
#         self.net_c = nn.Sequential(*[self._layer(features, style_dim) for _ in range(n_layers)])
#         self.net_s = nn.Sequential(*[self._layer(features, style_dim) for _ in range(n_layers)])

#     def _layer(self, features, style_dim):
#         return nn.Sequential(EqualizedLinear(features, features, style_dim=style_dim), nn.LeakyReLU(0.2, inplace=True))

#     def forward(self, z: torch.Tensor):
#         z = F.normalize(z, dim=2)  # Normalize across `in_features`
#         c_list, s_list = zip(*[(self.net_c(z[:, i, :]), self.net_s(z[:, i, :])) for i in range(z.size(1))])
#         return torch.stack(c_list, dim=1), torch.stack(s_list, dim=1)

# class MappingNetwork3D_cs_pyramid(nn.Module):
#     def __init__(self, features: int, n_layers: int, style_dim: int):
#         super().__init__()
#         self.coarse_ind, self.middle_ind = 3, 7
#         self.net_c = nn.Sequential(*[self._layer(features, style_dim) for _ in range(n_layers)])
#         self.net_s = nn.Sequential(*[self._layer(features, style_dim) for _ in range(n_layers)])

#     def _layer(self, features, style_dim):
#         return nn.Sequential(EqualizedLinear(features, features, style_dim=style_dim), nn.LeakyReLU(0.2, inplace=True))

#     def forward(self, z: torch.Tensor):
#         z = F.normalize(z, dim=2)  # Normalize across `in_features`
#         levels = [(0, self.coarse_ind), (self.coarse_ind, self.middle_ind), (self.middle_ind, z.size(1))]
#         c_parts, s_parts = zip(*[(self.net_c(z[:, start:end]), self.net_s(z[:, start:end])) for start, end in levels])
#         return torch.cat(c_parts, dim=1), torch.cat(s_parts, dim=1)
