from models.transformer.position_encoding import build_position_encoding
import torch
from torch import nn
import torch.nn.functional as F


class DETRTrackingBackbonePipeline(nn.Module):
    def __init__(self, backbone, position_encoding):
        super(DETRTrackingBackbonePipeline, self).__init__()
        self.backbone = backbone
        self.position_encoding = position_encoding
        assert len(self.backbone.num_channels_output) == 1
        self.num_channels = self.backbone.num_channels_output[0]

    def _forward_mask_pos(self, x_mask, x_feature):
        x_feature_mask = F.interpolate(x_mask.unsqueeze(dim=1).float(), size=x_feature.shape[-2:]).to(torch.bool).squeeze(dim=1)
        x_feature_pos = self.position_encoding(x_feature, x_feature_mask).to(x_feature.dtype)
        return x_feature_mask, x_feature_pos

    def forward(self, x, x_mask):
        x_feat = self.backbone(x)
        return (x_feat, *self._forward_mask_pos(x_mask, x_feat))


def build_backbone(net_config: dict):
    backbone_config = net_config['backbone']
    position_encoding = build_position_encoding(net_config)
    if backbone_config['type'] == 'alexnet':
        from models.backbone.pysot.alexnet import construct_alexnet
        backbone = construct_alexnet(output_layers=backbone_config['output_layers'])
    elif backbone_config['type'] == 'resnet50':
        from models.backbone.pysot.resnet_atrous import construct_resnet50_atrous
        backbone = construct_resnet50_atrous(output_layers=backbone_config['output_layers'])
    else:
        raise Exception(f'unsupported {backbone_config["type"]}')

    return DETRTrackingBackbonePipeline(backbone, position_encoding)
