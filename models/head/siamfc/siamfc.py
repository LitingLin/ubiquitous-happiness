import torch.nn as nn
import torch.nn.functional as F
import torch

__all__ = ['SiamFCLinearHead', 'SiamFCBNHead']


def _fast_xcorr(z, x):
    # fast cross correlation
    nz = z.size(0)
    nx, c, h, w = x.size()
    x = x.view(-1, nz * c, h, w)
    out = F.conv2d(x, z, groups=nz)
    out = out.view(nx, -1, out.size(-2), out.size(-1))
    return out


class SiamFCLinearHead(nn.Module):
    def __init__(self):
        super(SiamFCLinearHead, self).__init__()
        self.adjust_gain = nn.Parameter(torch.empty([1]))
        self.adjust_bias = nn.Parameter(torch.empty([1]))

    def reset_parameters(self):
        nn.init.constant_(self.adjust_gain.data, 0.001)
        nn.init.constant_(self.adjust_bias, 0.)

    def forward(self, z, x):
        return _fast_xcorr(z, x) * self.adjust_gain + self.adjust_bias


class SiamFCBNHead(nn.Module):
    def __init__(self, adjust_bn_eps=1e-5, adjust_bn_momentum=0.3):
        super(SiamFCBNHead, self).__init__()
        self.adjust_bn = nn.BatchNorm2d(1, eps=adjust_bn_eps, momentum=adjust_bn_momentum)

    def reset_parameters(self):
        nn.init.constant_(self.adjust_bn.weight, 1)
        nn.init.constant_(self.adjust_bn.bias, 0)

    def forward(self, z, x):
        return self.adjust_bn(_fast_xcorr(z, x))
