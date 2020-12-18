import torch
from torch import nn
import torch.nn.functional as F


class SiamFCBNConvHead(nn.Module):
    def __init__(self, n_channels):
        super(SiamFCBNConvHead, self).__init__()
        self.adjust_bn = nn.BatchNorm2d(n_channels)

    def reset_parameters(self):
        nn.init.constant_(self.adjust_bn.weight, 1)
        nn.init.constant_(self.adjust_bn.bias, 0)

    def forward(self, z, x):
        return self.adjust_bn(F.conv2d(x, z))
