import torch
import torch.nn as nn
from models.blocks.conv_bn_relu import conv_bn_relu
from models.operator.cross_correlation.func import xcorr_depthwise


class SiamFCDualPathXCorr(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, enable_cls_path=True, enable_reg_path=True):
        super(SiamFCDualPathXCorr, self).__init__()
        # feature adjustment
        if enable_cls_path:
            self.cls_z_conv = conv_bn_relu(input_dim, hidden_dim, 1, 3, 0, has_relu=False)
            self.cls_x_conv = conv_bn_relu(input_dim, hidden_dim, 1, 3, 0, has_relu=False)
        if enable_reg_path:
            self.reg_z_conv = conv_bn_relu(input_dim, hidden_dim, 1, 3, 0, has_relu=False)
            self.reg_x_conv = conv_bn_relu(input_dim, hidden_dim, 1, 3, 0, has_relu=False)
        self._initialize_weights()

        self.enable_cls_path = enable_cls_path
        self.enable_reg_path = enable_reg_path

    def _initialize_weights(self):
        def _init(m):
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight, std=0.01)

        self.apply(_init)

    def forward(self, z, x):
        if self.enable_cls_path:
            cls_z = self.cls_z_conv(z)
            cls_x = self.cls_x_conv(x)
            cls_out = xcorr_depthwise(cls_z, cls_x)
        else:
            cls_out = None
        if self.enable_reg_path:
            reg_z = self.reg_z_conv(z)
            reg_x = self.reg_x_conv(x)
            reg_out = xcorr_depthwise(reg_z, reg_x)
        else:
            reg_out = None

        return cls_out, reg_out
