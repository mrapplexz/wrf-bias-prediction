import numpy as np
from torch import nn
import torch
from timm.models.layers import DropPath

from correction.models.unet import UNet
from correction.models.utils import LayerNorm, GRN



class ConstantBias(nn.Module):
    def __init__(self, channels=6):
        super().__init__()
        self.channels = channels
        self.unet = UNet(4, 3)
        self.landmask_map = nn.Parameter(torch.tensor(np.load('landmask.npy')))
        self.landmask_map.requires_grad = False
        # self.spatial_map = nn.Parameter(torch.empty([2, 210, 280]))
        # nn.init.normal_(self.spatial_map)

    def forward(self, x_orig):
        '''
        input: S*B*C*H*W
        :param input:
        :return:
        '''
        x = torch.cat([
            x_orig,
            self.landmask_map.view(1, 1, 1, 210, 280).repeat(x_orig.shape[0], x_orig.shape[1], 1, 1, 1),
            # self.spatial_map.view(1, 1, 2, 210, 280).repeat(x.shape[0], x.shape[1], 1, 1, 1)
        ], dim=2)
        unet_out = self.unet(x)
        return x_orig + unet_out.view(x.shape[0], x.shape[1], 3, x.shape[3], x.shape[4])
