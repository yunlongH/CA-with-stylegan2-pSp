import torch
from torch import nn
# from models.encoders import psp_encoders
# from configs.paths_config import model_paths
from torch.nn import functional as F
# from models.stylegan2.model import Generator
from models.stylegan2.op import fused_leaky_relu
import math
import numpy as np

class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


class EqualLinear(nn.Module):
    def __init__(
            self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        self.activation = activation
        
        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

        self.lrelu = nn.LeakyReLU(self.lr_mul)

    def forward(self, input):
        if self.activation=='fused_lrelu':
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)

        elif self.activation=='fused_lrelu_normal':
            out = F.linear(input, self.weight * self.scale)
            out = self.lrelu(out + self.bias)
        
        else:
            out = F.linear(
                input, self.weight * self.scale, bias=self.bias * self.lr_mul
            )

        return out


class CS_Encoder_shared(nn.Module):
    def __init__(self, n_mlp=8, c_dim=512, s_dim=512, lr_mlp=0.01, cs_activation=None):
        super(CS_Encoder_shared, self).__init__()

        self.c_dim = c_dim
        self.s_dim = s_dim

        style_dim = c_dim + s_dim

        layers = [PixelNorm()]

        for i in range(n_mlp):
            layers.append(
                EqualLinear(
                    style_dim, style_dim, lr_mul=lr_mlp, activation=cs_activation#'fused_lrelu'
                )
            )

        self.style_net = nn.Sequential(*layers) 
        
    def forward(self, W):
        
        cat_w = torch.cat([W, W], dim=2)

        out = self.style_net(cat_w)

        latent_c = out[:,:, :self.c_dim]
        latent_s = out[:,:, self.s_dim:]

        return latent_c, latent_s

class CS_Encoder(nn.Module):
    def __init__(self, n_mlp=8, c_dim=512, s_dim=512, lr_mlp=0.01, cs_activation=None):
        super(CS_Encoder, self).__init__()

        self.c_dim = c_dim
        self.s_dim = s_dim

        style_dim = c_dim

        layers = [PixelNorm()]
        # layers_s = [PixelNorm()]

        for i in range(n_mlp):
            layers.append(
                EqualLinear(
                    style_dim, style_dim, lr_mul=lr_mlp, activation=cs_activation#'fused_lrelu'
                )
            )
            # layers.append(
            #     EqualLinear(
            #         style_dim, style_dim, lr_mul=lr_mlp, activation=None#'fused_lrelu'
            #     )
            # )

        self.style_c = nn.Sequential(*layers) 
        self.style_s = nn.Sequential(*layers) 

        
    def forward(self, W):
    
        latent_c = self.style_c(W)
        latent_s = self.style_s(W)
        # print('latent_c',latent_c.shape)
        # print('latent_s',latent_s.shape)
        return latent_c, latent_s



#

# Create the MLP
