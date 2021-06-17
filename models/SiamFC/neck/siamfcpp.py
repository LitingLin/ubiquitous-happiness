import torch
from torch import nn
import torch.nn.functional as F


class conv_bn_relu(nn.Module):
    def __init__(self,
                 in_channel,
                 out_channel,
                 stride=1,
                 kszie=3,
                 pad=0,
                 has_bn=True,
                 has_relu=True,
                 bias=True,
                 groups=1):
        r"""
        Basic block with one conv, one bn, one relu in series.
        Arguments
        ---------
        in_channel: int
            number of input channels
        out_channel: int
            number of output channels
        stride: int
            stride number
        kszie: int
            kernel size
        pad: int
            padding on each edge
        has_bn: bool
            use bn or not
        has_relu: bool
            use relu or not
        bias: bool
            conv has bias or not
        groups: int or str
            number of groups. To be forwarded to torch.nn.Conv2d
        """
        super(conv_bn_relu, self).__init__()
        self.conv = nn.Conv2d(in_channel,
                              out_channel,
                              kernel_size=kszie,
                              stride=stride,
                              padding=pad,
                              bias=bias,
                              groups=groups)

        if has_bn:
            self.bn = nn.BatchNorm2d(out_channel)
        else:
            self.bn = None

        if has_relu:
            self.relu = nn.ReLU()
        else:
            self.relu = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


def xcorr_depthwise(x, kernel):
    r"""
    Depthwise cross correlation. e.g. used for template matching in Siamese tracking network
    Arguments
    ---------
    x: torch.Tensor
        feature_x (N, C, H, W) (e.g. search region feature in SOT)
    kernel: torch.Tensor
        feature_z (N, C, H, W) (e.g. template feature in SOT)
    Returns
    -------
    torch.Tensor
        cross-correlation result
    """
    batch = int(kernel.size(0))
    channel = int(kernel.size(1))
    x = x.view(1, int(batch * channel), int(x.size(2)), int(x.size(3)))
    kernel = kernel.view(batch * channel, 1, int(kernel.size(2)),
                         int(kernel.size(3)))
    out = F.conv2d(x, kernel, groups=batch * channel)
    out = out.view(batch, channel, int(out.size(2)), int(out.size(3)))
    return out


class SiamFCPPNeck(nn.Module):
    def __init__(self, hidden_dim: int):
        super(SiamFCPPNeck, self).__init__()

        # feature adjustment
        self.r_z_k = conv_bn_relu(hidden_dim,
                                  hidden_dim,
                                  1,
                                  3,
                                  0,
                                  has_relu=False)
        self.c_z_k = conv_bn_relu(hidden_dim,
                                  hidden_dim,
                                  1,
                                  3,
                                  0,
                                  has_relu=False)
        self.r_x = conv_bn_relu(hidden_dim, hidden_dim, 1, 3, 0, has_relu=False)
        self.c_x = conv_bn_relu(hidden_dim, hidden_dim, 1, 3, 0, has_relu=False)

    def _initialize_weights(self):
        def _init(m):
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight, std=0.01)
        self.apply(_init)

    def forward(self, z, x):
        c_z_k = self.c_z_k(z)
        r_z_k = self.r_z_k(z)
        c_x = self.c_x(x)
        r_x = self.r_x(x)
        # feature matching
        r_out = xcorr_depthwise(r_x, r_z_k)
        c_out = xcorr_depthwise(c_x, c_z_k)
        return c_out, r_out

if __name__ == '__main__':
    import numpy as np
    def get_xy_ctr_np(score_size, score_offset, total_stride):
        """ generate coordinates on image plane for score map pixels (in numpy)
        """
        batch, fm_height, fm_width = 1, score_size, score_size

        y_list = np.linspace(0., fm_height - 1.,
                             fm_height).reshape(1, fm_height, 1, 1)
        y_list = y_list.repeat(fm_width, axis=2)
        x_list = np.linspace(0., fm_width - 1., fm_width).reshape(1, 1, fm_width, 1)
        x_list = x_list.repeat(fm_height, axis=1)
        xy_list = score_offset + np.concatenate((x_list, y_list), 3) * total_stride
        xy_ctr = np.repeat(xy_list, batch, axis=0).reshape(
            batch, -1,
            2)  # .broadcast([batch, fm_height, fm_width, 2]).reshape(batch, -1, 2)
        # TODO: consider use float32 type from the beginning of this function
        xy_ctr = torch.from_numpy(xy_ctr.astype(np.float32))
        return xy_ctr
    print(get_xy_ctr_np(7, 0, 8))