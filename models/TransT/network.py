import torch
from torch import nn


def _get_single_scale(feat):
    if isinstance(feat, (list, tuple)):
        assert len(feat) == 1
        feat = feat[0]
    return feat


def _get_backbone_output_channels(backbone):
    assert len(backbone.num_channels_output) == 1
    return backbone.num_channels_output[0]


class TransTTracking(nn.Module):
    def __init__(self, backbone, position_encoding, transformer, head):
        super().__init__()
        hidden_dim = transformer.d_model
        self.input_proj = nn.Conv2d(_get_backbone_output_channels(backbone), hidden_dim, kernel_size=1)

        self.backbone = backbone
        self.position_encoding = position_encoding
        self.transformer = transformer
        self.head = head

    def _forward_feat(self, x):
        x_feat = self.backbone(x)
        x_feat = _get_single_scale(x_feat)
        x_feat_pos = self.position_encoding(x_feat)
        x_feat = self.input_proj(x_feat)
        return x_feat, x_feat_pos

    def forward(self, z, x):
        z_feat, z_feat_pos = self._forward_feat(z)
        x_feat, x_feat_pos = self._forward_feat(x)

        hs = self.transformer(z_feat, x_feat, z_feat_pos, x_feat_pos)
        return self.head(hs)

    @torch.no_grad()
    def template(self, z):
        return self._forward_feat(z)

    @torch.no_grad()
    def track(self, z_feats, x):
        z_feat, z_feat_pos = z_feats
        x_feat, x_feat_pos = self._forward_feat(x)

        hs = self.transformer(z_feat, x_feat, z_feat_pos, x_feat_pos)
        return self.head(hs)
