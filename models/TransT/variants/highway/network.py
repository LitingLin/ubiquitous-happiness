import torch
from torch import nn


def _get_single_scale(feat):
    if isinstance(feat, (list, tuple)):
        assert len(feat) == 1
        feat = feat[0]
    return feat


def _forward_feat(x, backbone, position_encoder, input_proj):
    x_feat = backbone(x)
    x_feat = _get_single_scale(x_feat)
    x_feat_pos = position_encoder(x_feat)
    x_feat = input_proj(x_feat)
    return x_feat, x_feat_pos


def _get_backbone_output_channels(backbone):
    assert len(backbone.num_channels_output) == 1
    return backbone.num_channels_output[0]


class TransTTraskHighwayTracking(nn.Module):
    def __init__(self, backbone, classification_highway_backbone, regression_highway_backbone,
                 position_encoder, classification_position_encoder, regression_position_encoder,
                 transformer, classification_highway_transformer, regression_highway_transformer,
                 head):
        super().__init__()
        hidden_dim = transformer.d_model

        self.input_proj = nn.Conv2d(_get_backbone_output_channels(backbone), hidden_dim, kernel_size=1)
        self.classification_input_proj = nn.Conv2d(_get_backbone_output_channels(classification_highway_backbone), classification_highway_transformer.d_model, kernel_size=1)
        self.regression_input_proj = nn.Conv2d(_get_backbone_output_channels(regression_highway_backbone), regression_highway_transformer.d_model, kernel_size=1)

        self.backbone = backbone
        self.classification_highway_backbone = classification_highway_backbone
        self.regression_highway_backbone = regression_highway_backbone

        self.position_encoder = position_encoder
        self.classification_position_encoder = classification_position_encoder
        self.regression_position_encoder = regression_position_encoder

        self.transformer = transformer
        self.classification_highway_transformer = classification_highway_transformer
        self.regression_highway_transformer = regression_highway_transformer

        self.head = head

    def forward(self, z, x):
        z_feat, z_feat_pos = _forward_feat(z, self.backbone, self.position_encoder, self.input_proj)
        x_feat, x_feat_pos = _forward_feat(x, self.backbone, self.position_encoder, self.input_proj)

        hs = self.transformer(z_feat, x_feat, z_feat_pos, x_feat_pos)

        z_cls_feat, z_cls_feat_pos = _forward_feat(z, self.classification_highway_backbone,
                                                   self.classification_position_encoder, self.classification_input_proj)
        x_cls_feat, x_cls_feat_pos = _forward_feat(x, self.classification_highway_backbone,
                                                   self.classification_position_encoder, self.classification_input_proj)

        cls_hs = self.classification_highway_transformer(z_cls_feat, x_cls_feat, z_cls_feat_pos, x_cls_feat_pos)

        z_reg_feat, z_reg_feat_pos = _forward_feat(z, self.regression_highway_backbone,
                                                   self.regression_position_encoder, self.regression_input_proj)
        x_reg_feat, x_reg_feat_pos = _forward_feat(x, self.regression_highway_backbone,
                                                   self.regression_position_encoder, self.regression_input_proj)

        reg_hs = self.regression_highway_transformer(z_reg_feat, x_reg_feat, z_reg_feat_pos, x_reg_feat_pos)

        cls_hs = torch.cat((hs, cls_hs), dim=-1)
        reg_hs = torch.cat((hs, reg_hs), dim=-1)

        return self.head(cls_hs, reg_hs)

    @torch.no_grad()
    def template(self, z):
        z_feat, z_feat_pos = _forward_feat(z, self.backbone, self.position_encoder, self.input_proj)
        z_cls_feat, z_cls_feat_pos = _forward_feat(z, self.classification_highway_backbone,
                                                   self.classification_position_encoder, self.classification_input_proj)

        z_reg_feat, z_reg_feat_pos = _forward_feat(z, self.regression_highway_backbone,
                                                   self.regression_position_encoder, self.regression_input_proj)

        return z_feat, z_feat_pos, z_cls_feat, z_cls_feat_pos, z_reg_feat, z_reg_feat_pos

    @torch.no_grad()
    def track(self, z_feats, x):
        z_feat, z_feat_pos, z_cls_feat, z_cls_feat_pos, z_reg_feat, z_reg_feat_pos = z_feats
        x_feat, x_feat_pos = _forward_feat(x, self.backbone, self.position_encoder, self.input_proj)

        hs = self.transformer(z_feat, x_feat, z_feat_pos, x_feat_pos)

        x_cls_feat, x_cls_feat_pos = _forward_feat(x, self.classification_highway_backbone,
                                                   self.classification_position_encoder, self.classification_input_proj)

        cls_hs = self.classification_highway_transformer(z_cls_feat, x_cls_feat, z_cls_feat_pos, x_cls_feat_pos)

        x_reg_feat, x_reg_feat_pos = _forward_feat(x, self.regression_highway_backbone,
                                                   self.regression_position_encoder, self.regression_input_proj)

        reg_hs = self.regression_highway_transformer(z_reg_feat, x_reg_feat, z_reg_feat_pos, x_reg_feat_pos)

        cls_hs = torch.cat((hs, cls_hs), dim=-1)
        reg_hs = torch.cat((hs, reg_hs), dim=-1)

        return self.head(cls_hs, reg_hs)
