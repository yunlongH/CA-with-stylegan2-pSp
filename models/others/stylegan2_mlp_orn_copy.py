import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import List

class MappingNetwork(nn.Module):
    # """
    # <a id="mapping_network"></a>

    # ## Mapping Network

    # ![Mapping Network](mapping_network.svg)

    # This is an MLP with 8 linear layers.
    # The mapping network maps the latent vector $z \in \mathcal{W}$
    # to an intermediate latent space $w \in \mathcal{W}$.
    # $\mathcal{W}$ space will be disentangled from the image space
    # where the factors of variation become more linear.
    # """

    def __init__(self, features: int, n_layers: int):
        # """
        # * `features` is the number of features in $z$ and $w$
        # * `n_layers` is the number of layers in the mapping network.
        # """
        super().__init__()

        # Create the MLP
        layers = []
        for i in range(n_layers):
            # [Equalized learning-rate linear layers](#equalized_linear)
            layers.append(EqualizedLinear(features, features))
            # Leaky Relu
            layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))

        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor):
        # Normalize $z$
        z = F.normalize(z, dim=1)
        # Map $z$ to $w$
        return self.net(z)    


class MappingNetwork_cs(nn.Module):
    # """
    # <a id="mapping_network"></a>

    # ## Mapping Network

    # ![Mapping Network](mapping_network.svg)

    # This is an MLP with 8 linear layers.
    # The mapping network maps the latent vector $z \in \mathcal{W}$
    # to an intermediate latent space $w \in \mathcal{W}$.
    # $\mathcal{W}$ space will be disentangled from the image space
    # where the factors of variation become more linear.
    # """

    def __init__(self, features: int, n_layers: int):
        # """
        # * `features` is the number of features in $z$ and $w$
        # * `n_layers` is the number of layers in the mapping network.
        # """
        super().__init__()
        
        # Create the MLP
        layers_c = []
        layers_s = []

        for i in range(n_layers):
            
            layers_c.append(EqualizedLinear(features, features)) # [Equalized learning-rate linear layers](#equalized_linear)
            layers_c.append(nn.LeakyReLU(negative_slope=0.2, inplace=True)) # Leaky Relu

            layers_s.append(EqualizedLinear(features, features))
            layers_s.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))

        self.net_c = nn.Sequential(*layers_c)
        self.net_s = nn.Sequential(*layers_s)

    def forward(self, z: torch.Tensor):
        # Normalize $z$
        z = F.normalize(z, dim=1)

        # Map $z$ to $w$
        return self.net_c(z), self.net_s(z)
    
class MappingNetwork_cs_sparsity(nn.Module):

    def __init__(self, opts):
        # """
        # * `features` is the number of features in $z$ and $w$
        # * `n_layers` is the number of layers in the mapping network.
        # """
        super().__init__()

        features = opts.latent_dim
        n_layers = opts.n_layers_mlp

        self.opts = opts
        # Create the MLP
        layers_c = []
        layers_s = []

        for i in range(n_layers):
            
            layers_c.append(EqualizedLinear(features, features)) # [Equalized learning-rate linear layers](#equalized_linear)
            layers_c.append(nn.LeakyReLU(negative_slope=0.2, inplace=True)) # Leaky Relu
            layers_s.append(EqualizedLinear(features, features))
            layers_s.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))

        self.net_c = nn.Sequential(*layers_c)
        self.net_s = nn.Sequential(*layers_s)

    def zero_out(self, x):
        if self.opts.zero_out_type == 'hard':
            return torch.where(torch.abs(x) < self.opts.zero_threshold, torch.tensor(0.0, device=x.device), x)
        elif self.opts.zero_out_type == 'soft':
            return torch.sign(x) * torch.maximum(torch.abs(x) - self.opts.zero_threshold, torch.tensor(0.0, device=x.device))
        else:
            raise ValueError(f"Unknown zero out type: {self.opts.zero_out_type}. Use 'soft' or 'hard'.")

    def forward(self, z, zero_out_silent=False):
        # Normalize $z$
        z = F.normalize(z, dim=1)

        c = self.net_c(z)
        s = self.net_s(z)

        if zero_out_silent==True:
            #zero out silent of network
            s = self.zero_out(s)

        # Map $z$ to $w$
        return c, s
                

             


class MappingNetwork_cs_pos_out(nn.Module):
    # """
    # <a id="mapping_network"></a>

    # ## Mapping Network

    # ![Mapping Network](mapping_network.svg)

    # This is an MLP with 8 linear layers.
    # The mapping network maps the latent vector $z \in \mathcal{W}$
    # to an intermediate latent space $w \in \mathcal{W}$.
    # $\mathcal{W}$ space will be disentangled from the image space
    # where the factors of variation become more linear.
    # """

    def __init__(self, features: int, n_layers: int, act_fn: str, pos_s_only: bool):
        # """
        # * `features` is the number of features in $z$ and $w$
        # * `n_layers` is the number of layers in the mapping network.
        # """
        super().__init__()
        self.act_fn=act_fn
        # Create the MLP
        layers_c = []
        layers_s = []

        for i in range(n_layers-1):
            
            layers_c.append(EqualizedLinear(features, features)) # [Equalized learning-rate linear layers](#equalized_linear)
            layers_c.append(nn.LeakyReLU(negative_slope=0.2, inplace=True)) # Leaky Relu

            layers_s.append(EqualizedLinear(features, features))
            layers_s.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))

        
        layers_c.append(EqualizedLinear(features, features)) # [Equalized learning-rate linear layers](#equalized_linear)
        #layers_c.append(nn.LeakyReLU(negative_slope=0.2, inplace=True)) # Leaky Relu
        layers_s.append(EqualizedLinear(features, features))
        #layers_s.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        if pos_s_only==True:
            layers_c.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
            if act_fn=='sigmoid':
                print('using sigmoid s  activation')
                layers_s.append(nn.Sigmoid())

            elif act_fn=='relu':
                print('using relu s  activation')
                layers_s.append(nn.ReLU(inplace=True))

            else:
                print('using lrelu s  activation')
                layers_s.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))            
        else:
            if act_fn=='sigmoid':
                print('using sigmoid activation')
                layers_c.append(nn.Sigmoid())
                layers_s.append(nn.Sigmoid())

            elif act_fn=='relu':
                print('using relu activation')
                layers_c.append(nn.ReLU(inplace=True))
                layers_s.append(nn.ReLU(inplace=True))

            else:
                print('using lrelu activation')
                layers_c.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
                layers_s.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))

        self.net_c = nn.Sequential(*layers_c)
        self.net_s = nn.Sequential(*layers_s)

    def forward(self, z: torch.Tensor):
        # Normalize $z$
        z = F.normalize(z, dim=1)

        # c = self.net_c(z)
        # s = self.net_s(z)
        
        # Map $z$ to $w$
        return self.net_c(z), self.net_s(z)




class MappingNetwork_cs_individal(nn.Module):
    # """
    # <a id="mapping_network"></a>

    # ## Mapping Network

    # ![Mapping Network](mapping_network.svg)

    # This is an MLP with 8 linear layers.
    # The mapping network maps the latent vector $z \in \mathcal{W}$
    # to an intermediate latent space $w \in \mathcal{W}$.
    # $\mathcal{W}$ space will be disentangled from the image space
    # where the factors of variation become more linear.
    # """

    def __init__(self, features: int, n_layers: int):
        # """
        # * `features` is the number of features in $z$ and $w$
        # * `n_layers` is the number of layers in the mapping network.
        # """
        super().__init__()
        
        # Create the MLP
        layers_c = []
        layers_s = []

        for i in range(n_layers):
            
            layers_c.append(EqualizedLinear(features, features)) # [Equalized learning-rate linear layers](#equalized_linear)
            layers_c.append(nn.LeakyReLU(negative_slope=0.2, inplace=True)) # Leaky Relu

            layers_s.append(EqualizedLinear(features, features))
            layers_s.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))

        self.net_c = nn.Sequential(*layers_c)
        self.net_s = nn.Sequential(*layers_s)
    
    def tensor2styles(self, z):
        n_styles = z.shape[1]
        styles = []
        for i in range(n_styles):
            styles.append(z[:, i, :])
        return styles
    
    def styles2tensor2(self, styles):

        latent = torch.zeros(styles[0].shape[0], len(styles), styles[0].shape[1]).to(styles[0].device)

        for idx, s in enumerate(styles):
            latent[:, idx, :] = s

        return latent

    def forward(self, w: torch.Tensor):
        # Normalize $z$
        w = F.normalize(w, dim=1)   

        styles_w = self.tensor2styles(w)
        styles_c = []
        styles_s = []
        styles_c = [self.net_c(s) for s in styles_w]
        styles_s = [self.net_s(s) for s in styles_w]

        latent_c = self.styles2tensor2(styles_c)
        latent_s = self.styles2tensor2(styles_s)

        return latent_c, latent_s


class MappingNetwork_cs_sparsity(nn.Module):
    """
    Mapping Network with Controlled Sparsity in Outputs

    This network generates two mappings `c` and `s` from input `z`, with optional zeroing
    for sparsity based on the `zero_out_type` parameter ('hard' or 'soft'). This network
    uses MLP layers with learning-rate equalized weights.
    """

    def __init__(self, opts):
        """
        Initialize Mapping Network.

        Args:
            opts (object): Configuration options with attributes:
                - latent_dim (int): Dimensionality of latent vector `z`.
                - n_layers_mlp (int): Number of MLP layers in each network (`net_c` and `net_s`).
                - zero_out_type (str): Type of zeroing to apply ('hard' or 'soft').
                - zero_threshold (float): Threshold for zeroing out small values in output.
        
        Raises:
            ValueError: If `zero_out_type` is invalid or required options are missing.
        """
        super().__init__()

        # Validate required attributes in opts
        if not all(hasattr(opts, attr) for attr in ["latent_dim", "n_layers_mlp", "zero_out_type", "zero_threshold"]):
            raise ValueError("opts must contain 'latent_dim', 'n_layers_mlp', 'zero_out_type', and 'zero_threshold'.")

        features = opts.latent_dim
        n_layers = opts.n_layers_mlp
        self.opts = opts

        # Define MLP layers for each output (c and s)
        layers_c = []
        layers_s = []
        for i in range(n_layers):
            # [Equalized learning-rate linear layers](#equalized_linear)
            layers_c.append(EqualizedLinear(features, features))
            layers_c.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))  # Leaky ReLU activation
            
            layers_s.append(EqualizedLinear(features, features))
            layers_s.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))

        # Sequential networks for mapping `c` and `s`
        self.net_c = nn.Sequential(*layers_c)
        self.net_s = nn.Sequential(*layers_s)

    def zero_out(self, x):
        """
        Apply sparsity control to the output tensor `x` based on zero_out_type and zero_threshold.

        Args:
            x (torch.Tensor): Input tensor to apply zeroing.

        Returns:
            torch.Tensor: Tensor with values zeroed based on the specified zeroing method.
        """
        if self.opts.zero_out_type == 'hard':
            # Hard thresholding: values below `zero_threshold` are set to zero
            return torch.where(torch.abs(x) < self.opts.zero_threshold, torch.tensor(0.0, device=x.device), x)
        elif self.opts.zero_out_type == 'soft':
            # Soft thresholding: values are reduced towards zero smoothly
            return torch.sign(x) * torch.maximum(torch.abs(x) - self.opts.zero_threshold, torch.tensor(0.0, device=x.device))
        else:
            # Raise error if zero_out_type is unrecognized
            raise ValueError(f"Unknown zero out type: {self.opts.zero_out_type}. Use 'soft' or 'hard'.")

    def forward(self, z, zero_out_silent=False):
        """
        Forward pass through the mapping network.

        Args:
            z (torch.Tensor): Input latent vector of shape `(batch_size, latent_dim)`.
            zero_out_silent (bool): If `True`, apply zero_out to output `s` for controlled sparsity.

        Returns:
            tuple(torch.Tensor, torch.Tensor): Outputs `c` and `s`, representing two mappings of `z`.
        """
        # Normalize z for stability
        z = F.normalize(z, dim=1)

        # Process through each network
        c = self.net_c(z)
        s = self.net_s(z)

        # Conditionally apply zero_out to `s`
        if zero_out_silent:
            s = self.zero_out(s)

        # Return both outputs
        return c, s


class EqualizedLinear(nn.Module):
    """
    Learning-rate Equalized Linear Layer

    This layer uses an equalized learning rate approach, which normalizes weights
    to have a unit variance and then scales them by a constant `c`. It includes a learnable
    bias parameter.
    """

    def __init__(self, in_features: int, out_features: int, bias: float = 0.0):
        """
        Initialize the Equalized Linear layer.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            bias (float): Initial bias value (default: 0.0).
        """
        super().__init__()

        # Initialize learning-rate equalized weight parameter
        self.weight = EqualizedWeight([out_features, in_features])

        # Initialize bias as a learnable parameter
        self.bias = nn.Parameter(torch.ones(out_features) * bias)

    def forward(self, x: torch.Tensor):
        """
        Forward pass with learning-rate equalized linear transformation.

        Args:
            x (torch.Tensor): Input tensor of shape `(batch_size, in_features)`.

        Returns:
            torch.Tensor: Output tensor of shape `(batch_size, out_features)`.
        """
        # Apply the linear transformation with equalized weights and bias
        return F.linear(x, self.weight(), bias=self.bias)


class EqualizedWeight(nn.Module):
    """
    Learning-rate Equalized Weight Parameter

    Implements learning-rate equalization by scaling weights dynamically based on a constant `c`.
    This technique normalizes the weights to have unit variance initially, then scales by `c`
    for consistent gradient flow and effective learning rates.
    """

    def __init__(self, shape: List[int]):
        """
        Initialize the Equalized Weight parameter.

        Args:
            shape (List[int]): Shape of the weight parameter tensor.
        
        Attributes:
            c (float): Scaling constant based on He initialization.
            weight (torch.Parameter): Initialized weight parameter with shape `shape`.
        """
        super().__init__()

        # Calculate scaling constant `c` based on He initialization
        self.c = 1 / math.sqrt(np.prod(shape[1:]))

        # Initialize weight with a normal distribution `N(0, 1)`
        self.weight = nn.Parameter(torch.randn(shape))

    def forward(self):
        """
        Scaled weight forward pass.

        Returns:
            torch.Tensor: Weight tensor scaled by the constant `c`.
        """
        # Scale the weight by constant `c` and return
        return self.weight * self.c


class MappingNetwork_cs_individal2(nn.Module):
    # """
    # <a id="mapping_network"></a>

    # ## Mapping Network

    # ![Mapping Network](mapping_network.svg)

    # This is an MLP with 8 linear layers.
    # The mapping network maps the latent vector $z \in \mathcal{W}$
    # to an intermediate latent space $w \in \mathcal{W}$.
    # $\mathcal{W}$ space will be disentangled from the image space
    # where the factors of variation become more linear.
    # """

    def __init__(self, features: int, n_layers: int):
        # """
        # * `features` is the number of features in $z$ and $w$
        # * `n_layers` is the number of layers in the mapping network.
        # """
        super().__init__()
        # self.style_count = 18
        self.coarse_ind = 3
        self.middle_ind = 7
        
        # Create the MLP
        layers_c = []
        layers_s = []

        for i in range(n_layers):
            
            layers_c.append(EqualizedLinear(features, features)) # [Equalized learning-rate linear layers](#equalized_linear)
            layers_c.append(nn.LeakyReLU(negative_slope=0.2, inplace=True)) # Leaky Relu

            layers_s.append(EqualizedLinear(features, features))
            layers_s.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))

        self.net_c = nn.Sequential(*layers_c)
        self.net_s = nn.Sequential(*layers_s)

    def forward(self, z: torch.Tensor):
        # Normalize $z$
        z = F.normalize(z, dim=1)

        c_list=[]
        s_list=[]
        
        for i in range (z.shape[1]):
            c_list.append(self.net_c(z[:,i,:]))
            s_list.append(self.net_s(z[:,i,:]))

        latent_c = torch.stack(c_list, dim=1)
        latent_s = torch.stack(s_list, dim=1)

        # Map $z$ to $w$
        return latent_c, latent_s

class MappingNetwork_cs_pyramid(nn.Module):
    # """
    # <a id="mapping_network"></a>

    # ## Mapping Network

    # ![Mapping Network](mapping_network.svg)

    # This is an MLP with 8 linear layers.
    # The mapping network maps the latent vector $z \in \mathcal{W}$
    # to an intermediate latent space $w \in \mathcal{W}$.
    # $\mathcal{W}$ space will be disentangled from the image space
    # where the factors of variation become more linear.
    # """

    def __init__(self, features: int, n_layers: int):
        # """
        # * `features` is the number of features in $z$ and $w$
        # * `n_layers` is the number of layers in the mapping network.
        # """
        super().__init__()
        # self.style_count = 18
        self.coarse_ind = 3
        self.middle_ind = 7

        # Create the MLP
        layers_c = []
        layers_s = []

        for i in range(n_layers):
            
            layers_c.append(EqualizedLinear(features, features)) # [Equalized learning-rate linear layers](#equalized_linear)
            layers_c.append(nn.LeakyReLU(negative_slope=0.2, inplace=True)) # Leaky Relu

            layers_s.append(EqualizedLinear(features, features))
            layers_s.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))

        self.net_c = nn.Sequential(*layers_c)
        self.net_s = nn.Sequential(*layers_s)

    def forward(self, z: torch.Tensor):
        # Normalize $z$
        z = F.normalize(z, dim=1)

        z_lv1 = z[:, :self.coarse_ind, :]
        z_lv2 = z[:, self.coarse_ind:self.middle_ind, :]
        z_lv3 = z[:, self.middle_ind:, :]

        c_lv1 = self.net_c(z_lv1)
        s_lv1 = self.net_s(z_lv1)

        c_lv2 = self.net_c(z_lv2)
        s_lv2 = self.net_s(z_lv2)

        c_lv3 = self.net_c(z_lv3)
        s_lv3 = self.net_s(z_lv3)

        #print(len(latents))
        latent_c = torch.cat([c_lv1, c_lv2, c_lv3], dim=1)
        latent_s = torch.cat([s_lv1, s_lv2, s_lv3], dim=1)
        
        # Map $z$ to $w$
        return latent_c, latent_s

