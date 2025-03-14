import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Linear, Conv2d, BatchNorm2d, PReLU, Sequential, Module

from models.encoders.helpers import get_blocks, Flatten, bottleneck_IR, bottleneck_IR_SE
from models.stylegan2.model import EqualLinear


class GradualStyleBlock(Module):
    def __init__(self, in_c, out_c, spatial):
        super(GradualStyleBlock, self).__init__()
        self.out_c = out_c
        self.spatial = spatial
        num_pools = int(np.log2(spatial))
        modules = []
        modules += [Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1),
                    nn.LeakyReLU()]
        for i in range(num_pools - 1):
            modules += [
                Conv2d(out_c, out_c, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU()
            ]
        self.convs = nn.Sequential(*modules)
        self.linear = EqualLinear(out_c, out_c, lr_mul=1)

    def forward(self, x):
        x = self.convs(x)

        x = x.view(-1, self.out_c)

        x = self.linear(x)

        return x



class GradualConstructiveStyleEncoder(Module):
    def __init__(self, num_layers, mode='ir', opts=None):
        super(GradualConstructiveStyleEncoder, self).__init__()
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(opts.input_nc, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)

        self.styles_c = nn.ModuleList()
        self.styles_s = nn.ModuleList()
        self.style_count = opts.n_styles
        self.coarse_ind = 3
        self.middle_ind = 7
        for i in range(self.style_count):
            if i < self.coarse_ind:
                style_c = GradualStyleBlock(512, 512, 16)
            elif i < self.middle_ind:
                style_c = GradualStyleBlock(512, 512, 32)
            else:
                style_c = GradualStyleBlock(512, 512, 64)
            self.styles_c.append(style_c)

        for i in range(self.style_count):
            if i < self.coarse_ind:
                style_s = GradualStyleBlock(512, 512, 16)
            elif i < self.middle_ind:
                style_s = GradualStyleBlock(512, 512, 32)
            else:
                style_s = GradualStyleBlock(512, 512, 64)
            self.styles_s.append(style_s)

        self.latlayer1 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0)

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y

    def forward(self, x):
        x = self.input_layer(x)
        #print('x shape', x.shape)

        latents_c = []
        latents_s = []
        modulelist = list(self.body._modules.values())
        for i, l in enumerate(modulelist):
            x = l(x)
            if i == 6:
                c1 = x
            elif i == 20:
                c2 = x
            elif i == 23:
                c3 = x

        # print('c1 shape',c1.shape)
        # print('c2 shape',c2.shape)
        # print('c3 shape',c3.shape)

        for j in range(self.coarse_ind):
            latents_c.append(self.styles_c[j](c3))
            latents_s.append(self.styles_s[j](c3))

        #print(len(latents))

        p2 = self._upsample_add(c3, self.latlayer1(c2))
        for j in range(self.coarse_ind, self.middle_ind):
            latents_c.append(self.styles_c[j](p2))
            latents_s.append(self.styles_s[j](p2))

        #print(len(latents))

        p1 = self._upsample_add(p2, self.latlayer2(c1))
        for j in range(self.middle_ind, self.style_count):
            latents_c.append(self.styles_c[j](p1))
            latents_s.append(self.styles_s[j](p1))

        #print(len(latents))
        out_c = torch.stack(latents_c, dim=1)
        out_s = torch.stack(latents_s, dim=1)
        #print('outshape ',out.shape)
        #print(out.shape)
        return out_c, out_s
