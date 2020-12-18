import torch
from torch import nn
import torch.nn.functional as F


def _cross_correlation(z, x):
    z_n, z_c, z_h, z_w = z.shape
    x_n, x_c, x_h, x_w = x.shape
    assert z_n == x_n and z_c == x_c
    x = x.view(1, x_n * x_c, x_h, x_w)
    z = z.view(z_n * z_c, 1, z_h, z_w)
    res = F.conv2d(x, z, groups=x_n * x_c)
    res_h, res_w = res.shape[2:4]
    return res.view(z_n, z_c, res_h, res_w)


class SiamFCBNConvHead(nn.Module):
    def __init__(self, n_channels):
        super(SiamFCBNConvHead, self).__init__()
        self.adjust_bn = nn.BatchNorm2d(n_channels)

    def reset_parameters(self):
        nn.init.constant_(self.adjust_bn.weight, 1)
        nn.init.constant_(self.adjust_bn.bias, 0)

    def forward(self, z, x):
        return self.adjust_bn(_cross_correlation(z, x))
