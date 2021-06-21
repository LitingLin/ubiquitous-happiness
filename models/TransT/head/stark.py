import torch
from torch import nn
import torch.nn.functional
from models.modules.frozen_batch_norm import FrozenBatchNorm2d
from models.modules.mlp import MLP
from data.operator.bbox.spatial.vectorized.torch.xyxy_to_cxcywh import box_xyxy_to_cxcywh


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1,
         freeze_bn=False):
    if freeze_bn:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=True),
            FrozenBatchNorm2d(out_planes),
            nn.ReLU(inplace=True))
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=True),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True))


class Corner_Predictor(nn.Module):
    """ Corner Predictor module"""
    def __init__(self, inplanes=64, channel=256, feat_sz=20, freeze_bn=False):
        super(Corner_Predictor, self).__init__()
        self.feat_sz = feat_sz
        '''top-left corner'''
        self.conv1_tl = conv(inplanes, channel, freeze_bn=freeze_bn)
        self.conv2_tl = conv(channel, channel // 2, freeze_bn=freeze_bn)
        self.conv3_tl = conv(channel // 2, channel // 4, freeze_bn=freeze_bn)
        self.conv4_tl = conv(channel // 4, channel // 8, freeze_bn=freeze_bn)
        self.conv5_tl = nn.Conv2d(channel // 8, 1, kernel_size=1)

        '''bottom-right corner'''
        self.conv1_br = conv(inplanes, channel, freeze_bn=freeze_bn)
        self.conv2_br = conv(channel, channel // 2, freeze_bn=freeze_bn)
        self.conv3_br = conv(channel // 2, channel // 4, freeze_bn=freeze_bn)
        self.conv4_br = conv(channel // 4, channel // 8, freeze_bn=freeze_bn)
        self.conv5_br = nn.Conv2d(channel // 8, 1, kernel_size=1)

        indices = torch.linspace(0, 1, self.feat_sz)
        self.register_buffer('coord_x', indices.repeat((self.feat_sz, 1)).view((self.feat_sz * self.feat_sz,)))
        self.register_buffer('coord_y', indices.repeat((1, self.feat_sz)).view((self.feat_sz * self.feat_sz,)))

    def forward(self, x):
        """ Forward pass with input x. """
        score_map_tl, score_map_br = self.get_score_map(x)
        coorx_tl, coory_tl = self.soft_argmax(score_map_tl)
        coorx_br, coory_br = self.soft_argmax(score_map_br)
        return torch.stack((coorx_tl, coory_tl, coorx_br, coory_br), dim=1)

    def get_score_map(self, x):
        # top-left branch
        x_tl1 = self.conv1_tl(x)
        x_tl2 = self.conv2_tl(x_tl1)
        x_tl3 = self.conv3_tl(x_tl2)
        x_tl4 = self.conv4_tl(x_tl3)
        score_map_tl = self.conv5_tl(x_tl4)

        # bottom-right branch
        x_br1 = self.conv1_br(x)
        x_br2 = self.conv2_br(x_br1)
        x_br3 = self.conv3_br(x_br2)
        x_br4 = self.conv4_br(x_br3)
        score_map_br = self.conv5_br(x_br4)
        return score_map_tl, score_map_br

    def soft_argmax(self, score_map):
        """ get soft-argmax coordinate for a given heatmap """
        prob_vec = nn.functional.softmax(
            score_map.view((-1, self.feat_sz * self.feat_sz)), dim=1)  # (batch, feat_sz * feat_sz)
        exp_x = torch.sum((self.coord_x * prob_vec), dim=1)
        exp_y = torch.sum((self.coord_y * prob_vec), dim=1)
        return exp_x, exp_y


class Corner_Predictor_MLP(nn.Module):
    """ Corner Predictor module"""
    def __init__(self, inplanes=64, channel=256, feat_sz=20):
        super(Corner_Predictor_MLP, self).__init__()
        self.feat_sz = feat_sz
        '''top-left corner'''
        self.tl_mlp = MLP(inplanes, channel, 1, 5)
        '''bottom-right corner'''
        self.br_mlp = MLP(inplanes, channel, 1, 5)

        indices = torch.linspace(0, 1, self.feat_sz)
        self.register_buffer('coord_x', indices.repeat((self.feat_sz, 1)).view((self.feat_sz * self.feat_sz,)))
        self.register_buffer('coord_y', indices.repeat((1, self.feat_sz)).view((self.feat_sz * self.feat_sz,)))

    def forward(self, x):
        """
            Forward pass with input x.
            x: torch.Tensor
                (N, L, C)
        """
        score_map_tl, score_map_br = self.get_score_map(x)
        coorx_tl, coory_tl = self.soft_argmax(score_map_tl)
        coorx_br, coory_br = self.soft_argmax(score_map_br)
        return torch.stack((coorx_tl, coory_tl, coorx_br, coory_br), dim=1)

    def get_score_map(self, x):
        N = x.size(0)
        # top-left branch
        score_map_tl = self.tl_mlp(x)
        score_map_tl = score_map_tl.permute(1, 2).reshape(N, H, W)

        # bottom-right branch
        score_map_br = self.br_mlp(x)
        score_map_br = score_map_br.permute(1, 2).reshape(N, H, W)
        return score_map_tl, score_map_br

    def soft_argmax(self, score_map):
        """ get soft-argmax coordinate for a given heatmap """
        prob_vec = nn.functional.softmax(
            score_map.view((-1, self.feat_sz * self.feat_sz)), dim=1)  # (batch, feat_sz * feat_sz)
        exp_x = torch.sum((self.coord_x * prob_vec), dim=1)
        exp_y = torch.sum((self.coord_y * prob_vec), dim=1)
        return exp_x, exp_y


class StarkHead:
    def __init__(self, feature_map_size, hidden_dim):
        self.classification_branch = MLP(hidden_dim, hidden_dim, 1, 3)
        self.localization_branch = Corner_Predictor(inplanes=hidden_dim, channel=256, feat_sz=feature_map_size)
        self.feature_map_size = feature_map_size

    def forward(self, x):
        '''
        Args:
            x (torch.Tensor): (S, N, L, C)
        '''
        assert x.shape[0] == 1
        x = x[0]

        cls = self.classification_branch(x)
        x = x.transpose(1, 2)
        x = x.view(x.shape[0], x.shape[1], self.feature_map_size, self.feature_map_size)
        box = self.localization_branch(x)
        box = box_xyxy_to_cxcywh(box)
        return cls, box


class StarkTransformerHead:
    def __init__(self, feature_map_size, classification_input_dim, classification_hidden_dim,
                 localization_input_dim, localization_hidden_dim,
                 enable_classification = True, enable_localization = True):
        if enable_classification:
            pass


class StarkSHead(nn.Module):
    def __init__(self, localization_branch):
        super(StarkSHead, self).__init__()
        self.localization_branch = localization_branch

    def forward(self, x):
        '''
        Args:
            x (torch.Tensor): (S, N, L, C)
        '''
        assert x.shape[0] == 1, 'Only support one scale currently'
        x = x[0]

        box = self.localization_branch(x)
        box = box_xyxy_to_cxcywh(box)
        return box


def build_stark_head(network_config: dict):
    assert network_config['head']['type'] == 'Stark'
    head_parameters = network_config['head']['parameters']

    input_dim = head_parameters['input_dim']
    feature_map_size = network_config['data']['feature_size']

    if head_parameters['sub_type'] == 'StarkS':
        localization_branch_type = head_parameters['localization']['type']
        localization_branch_parameters = head_parameters['localization']['parameters']

        hidden_dim = localization_branch_parameters['hidden_dim']

        if localization_branch_type == 'CornerPredictor':
            return StarkSHead(Corner_Predictor(input_dim, hidden_dim, feature_map_size))
        elif localization_branch_type == 'CornerPredictorMLP':
            return StarkSHead(Corner_Predictor_MLP(input_dim, hidden_dim, feature_map_size))
        else:
            raise NotImplementedError(f'Unknown localization branch type: {localization_branch_type}')
    else:
        raise NotImplementedError(f'Unknown head sub type: {head_parameters["sub_type"]}')
