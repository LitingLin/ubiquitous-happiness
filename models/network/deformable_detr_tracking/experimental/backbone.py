from models.transformer.position_encoding import build_position_encoding
import torch
from torch import nn
import torch.nn.functional as F


class DeformDETRTrackBackbonePipeline(nn.Module):
    def __init__(self, backbone, position_encoding, z_position_encoding_offset=5000):
        super(DeformDETRTrackBackbonePipeline, self).__init__()
        self.backbone = backbone
        self.position_encoding = position_encoding
        self.z_position_encoding_offset = z_position_encoding_offset

    def _forward_mask_pos(self, x_mask, x_feature, pos_offset):
        x_feature_mask = F.interpolate(x_mask.unsqueeze(dim=1).float(), size=x_feature.shape[-2:]).to(torch.bool).squeeze(dim=1)
        x_feature_pos = self.position_encoding(x_feature, x_feature_mask, pos_offset).to(x_feature.dtype)
        return x_feature_mask, x_feature_pos

    def _forward(self, x, x_mask, pos_offset):
        x_feat = self.backbone(x)

        if isinstance(x_feat, list):
            mask = []
            pos = []
            for feature_level in x_feat:
                feat_mask, feat_pos = self._forward_mask_pos(x_mask, feature_level, pos_offset)
                mask.append(feat_mask)
                pos.append(feat_pos)
        else:
            mask, pos = self._forward_mask_pos(x_mask, x_feat, pos_offset)
        return x_feat, mask, pos

    def forward(self, x, x_mask, z, z_mask):
        x_feat, x_feat_mask, x_feat_pos = self._forward(x, x_mask, 0)
        z_feat, z_feat_mask, z_feat_pos = self._forward(z, z_mask, self.z_position_encoding_offset)
        return x_feat, x_feat_mask, x_feat_pos, z_feat, z_feat_mask, z_feat_pos


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

    return DeformDETRTrackBackbonePipeline(backbone, position_encoding, backbone_config['z_position_encoding_offset'])
