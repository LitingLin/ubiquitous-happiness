import torch
from torch import nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, act_fn=F.relu):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.act_fn = act_fn

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act_fn(layer(x))
        return x

def _init_linear_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=.02)
        nn.init.constant_(m.bias, 0)


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
        self.shared_mlp = MLP(input_dim, hidden_dim, hidden_dim, 2)
        self.classification_head = nn.Linear(hidden_dim, 1)
        self.regression_head = nn.Linear(hidden_dim, 4 * (gfocal_reg_max + 1))

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
        self.apply(_init_linear_weights)

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
        shared_feat = self.shared_mlp(x)

        reg_pred = self.regression_head(shared_feat)  # (N, H * W, 4 * (gfocal_reg_max + 1))
        prob = F.softmax(reg_pred.transpose(1, 2).reshape(N, 4, self.gfocal_reg_max + 1, H, W), dim=2)

        prob_topk, _ = prob.topk(self.gfocal_v2_topk, dim=2)
        if self.gfocal_v2_add_mean:
            stat = torch.cat([prob_topk, prob_topk.mean(dim=2, keepdim=True)],
                             dim=2)
        else:
            stat = prob_topk

        quality_score = self.reg_conf(stat.reshape(N, -1, H, W))

        cls_score = self.classification_head(shared_feat).transpose(1, 2).reshape(N, -1, H, W)
        cls_score.sigmoid_()
        cls_score = cls_score * quality_score

        return cls_score, self.integral(reg_pred).view(N, H, W, 4) / self.gfocal_reg_max, reg_pred.view(N, H, W, 4 * (self.gfocal_reg_max + 1))
