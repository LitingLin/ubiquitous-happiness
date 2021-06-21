import torch
from torch import nn
from models.blocks.conv_bn_relu import conv_bn_relu
from models.operator.cross_correlation.func import xcorr_depthwise


class SiamFCXCorr(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super(SiamFCXCorr, self).__init__()

        # feature adjustment
        self.z_conv = conv_bn_relu(input_dim, hidden_dim, 1, 3, 0, has_relu=False)
        self.x_conv = conv_bn_relu(input_dim, hidden_dim, 1, 3, 0, has_relu=False)
        self._initialize_weights()

    def _initialize_weights(self):
        def _init(m):
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight, std=0.01)

        self.apply(_init)

    def forward(self, z, x):
        z = self.z_conv(z)
        x = self.x_conv(x)
        # feature matching
        out = xcorr_depthwise(z, x)
        return out
