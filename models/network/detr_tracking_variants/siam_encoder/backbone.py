import torch
from torch import nn
import torch.nn.functional as F


# generate feat mask and pos encoding
class BackboneMaskWrapper(nn.Module):
    def __init__(self, backbone, position_encoding):
        super(BackboneMaskWrapper, self).__init__()

        self.backbone = backbone
        self.position_encoding = position_encoding

    def _forward_mask_pos(self, input_mask, input_feature):
        x_feature_mask = F.interpolate(input_mask.unsqueeze(dim=1).float(), size=input_feature.shape[-2:]).to(torch.bool).squeeze(dim=1)
        x_feature_pos = self.position_encoding(input_feature, x_feature_mask).to(input_feature.dtype)
        return x_feature_mask, x_feature_pos

    def forward(self, input_, input_mask):
        input_feat = self.backbone(input_)
        input_feat_mask, input_feat_pos = self._forward_mask_pos(input_mask, input_feat)
        return input_feat, input_feat_mask, input_feat_pos


# generate feat mask and pos encoding
class SiamEncoderBackboneMaskWrapper(nn.Module):
    def __init__(self, backbone, position_encoding):
        super(SiamEncoderBackboneMaskWrapper, self).__init__()
        self.backbone = backbone
        self.position_encoding = position_encoding
        self.num_channels = self.backbone.num_channels

    def _forward_z_mask_pos(self, z_mask, z_feature):
        z_feature_mask = F.interpolate(z_mask.unsqueeze(dim=1).float(), size=z_feature.shape[-2:]).to(torch.bool).squeeze(dim=1)
        z_feature_pos = self.position_encoding(z_feature, z_feature_mask)
        return z_feature_mask, z_feature_pos

    def _forward_x_mask_pos(self, x_feature):
        n, c, h, w = x_feature.shape
        x_feat_mask = torch.zeros((n, h, w), dtype=torch.bool, device=x_feature.device)
        x_feat_pos = self.position_encoding(x_feature, x_feat_mask)
        return x_feat_mask, x_feat_pos

    def forward(self, z, z_mask, x):
        z_feat = self.backbone(z)
        z_feat_mask, z_feat_pos = self._forward_z_mask_pos(z_mask, z_feat)

        x_feat = self.backbone(x)
        x_feat_mask, x_feat_pos = self._forward_x_mask_pos(x_feat)

        return z_feat, z_feat_mask, z_feat_pos, x_feat, x_feat_mask, x_feat_pos
