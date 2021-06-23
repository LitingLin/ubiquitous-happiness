import torch.nn as nn
import torch
from models.operator.cross_correlation.func import xcorr

__all__ = ['SiamFCLinearNeck', 'SiamFCBNNeck']


class SiamFCLinearNeck(nn.Module):
    def __init__(self):
        super(SiamFCLinearNeck, self).__init__()
        self.adjust_gain = nn.Parameter(torch.empty([1]))
        self.adjust_bias = nn.Parameter(torch.empty([1]))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.adjust_gain.data, 0.001)
        nn.init.constant_(self.adjust_bias, 0.)

    def forward(self, z, x):
        return xcorr(z, x) * self.adjust_gain + self.adjust_bias


class SiamFCBNNeck(nn.Module):
    def __init__(self, adjust_bn_eps=1e-5, adjust_bn_momentum=0.3):
        super(SiamFCBNNeck, self).__init__()
        self.adjust_bn = nn.BatchNorm2d(1, eps=adjust_bn_eps, momentum=adjust_bn_momentum)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.adjust_bn.weight, 1)
        nn.init.constant_(self.adjust_bn.bias, 0)

    def forward(self, z, x):
        return self.adjust_bn(xcorr(z, x))
