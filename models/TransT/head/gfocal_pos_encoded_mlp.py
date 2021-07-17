import torch
import torch.nn.functional as F
import torch.nn as nn
from timm.models.layers import to_2tuple, trunc_normal_
import math


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, to_2tuple(3), to_2tuple(1), 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv1 = DWConv(hidden_features)
        self.fc2 = nn.Linear(hidden_features, hidden_features)
        self.dwconv2 = DWConv(hidden_features)
        self.fc3 = nn.Linear(hidden_features, out_features)

        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv1(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.dwconv2(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc3(x)
        x = self.drop(x)
        return x


class Integral(nn.Module):
    """A fixed layer for calculating integral result from distribution.

    This layer calculates the target location by :math: `sum{P(y_i) * y_i}`,
    P(y_i) denotes the softmax vector that represents the discrete distribution
    y_i denotes the discrete set, usually {0, 1, 2, ..., reg_max}

    Args:
        reg_max (int): The maximal value of the discrete set. Default: 16. You
            may want to reset it according to your new dataset or related
            settings.
    """

    def __init__(self, reg_max=16):
        super(Integral, self).__init__()
        self.reg_max = reg_max
        self.register_buffer('project',
                             torch.linspace(0, self.reg_max, self.reg_max + 1))

    def forward(self, x):
        """Forward feature from the regression head to get integral result of
        bounding box location.

        Args:
            x (Tensor): Features of the regression head, shape (N, 4*(n+1)),
                n is self.reg_max.

        Returns:
            x (Tensor): Integral result of box locations, i.e., distance
                offsets from the box center in four directions, shape (N, 4).
        """
        x = F.softmax(x.reshape(-1, self.reg_max + 1), dim=1)
        x = F.linear(x, self.project.type_as(x)).reshape(-1, 4)
        return x


class GFocalV2Head(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, shape,
                 gfocal_reg_max: int,  # bbox in range [0, reg_max]
                 gfocal_v2_topk: int, gfocal_v2_reg_channels: int, gfocal_v2_add_mean: bool):
        super(GFocalV2Head, self).__init__()
        self.classification_mlp = Mlp(input_dim, hidden_dim, 1)
        self.regression_mlp = Mlp(input_dim, hidden_dim, 4 * (gfocal_reg_max + 1))

        self.shape = shape  # (W, H)
        self.integral = Integral(gfocal_reg_max)
        self.gfocal_reg_max = gfocal_reg_max
        self.gfocal_v2_topk = gfocal_v2_topk
        self.gfocal_v2_reg_channels = gfocal_v2_reg_channels
        self.gfocal_v2_add_mean = gfocal_v2_add_mean
        self.gfocal_v2_total_dim = gfocal_v2_topk
        if gfocal_v2_add_mean:
            self.gfocal_v2_total_dim += 1

        self.reg_conf = nn.Sequential(
            nn.Conv2d(4 * self.gfocal_v2_total_dim, self.gfocal_v2_reg_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.gfocal_v2_reg_channels, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        '''
        Args:
            x: (torch.Tensor): (S, N, L, C)
        '''
        assert x.shape[0] == 1
        x = x[0]
        N = x.shape[0]
        H = self.shape[1]
        W = self.shape[0]

        reg_pred = self.regression_mlp(x, H, W)  # (N, H * W, 4 * (gfocal_reg_max + 1))
        prob = F.softmax(reg_pred.transpose(1, 2).reshape(N, 4, self.gfocal_reg_max + 1, H, W), dim=2)

        prob_topk, _ = prob.topk(self.gfocal_v2_topk, dim=2)
        if self.gfocal_v2_add_mean:
            stat = torch.cat([prob_topk, prob_topk.mean(dim=2, keepdim=True)],
                             dim=2)
        else:
            stat = prob_topk

        quality_score = self.reg_conf(stat.reshape(N, -1, H, W))

        cls_score = self.classification_mlp(x, H, W).transpose(1, 2).reshape(N, -1, H, W)
        cls_score.sigmoid_()
        cls_score = cls_score * quality_score

        return cls_score, self.integral(reg_pred).view(N, H, W, 4) / self.gfocal_reg_max, reg_pred.view(N, H, W, 4 * (self.gfocal_reg_max + 1))
