import torch
from torch import nn

from models.modules.mlp import MLP


class TransTTracking(nn.Module):
    def __init__(self, backbone, transformer):
        super().__init__()
        hidden_dim = transformer.d_model

        self.class_embed = MLP(hidden_dim, hidden_dim, 2, 3)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)

        self.backbone = backbone
        self.transformer = transformer

    def forward(self, input_):
        z, x = input_
        z_feat, z_feat_pos = self.backbone(z)
        x_feat, x_feat_pos = self.backbone(x)

        z_feat = self.input_proj(z_feat)
        x_feat = self.input_proj(x_feat)

        hs = self.transformer(z_feat, x_feat, z_feat_pos, x_feat_pos)

        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()

        return outputs_class[-1], outputs_coord[-1]

    @torch.no_grad()
    def template(self, z):
        z_feat, z_feat_pos = self.backbone(z)
        z_feat = self.input_proj(z_feat)
        return z_feat, z_feat_pos

    @torch.no_grad()
    def track(self, z_feats, x):
        z_feat, z_feat_pos = z_feats
        x_feat, x_feat_pos = self.backbone(x)
        x_feat = self.input_proj(x_feat)

        hs = self.transformer(z_feat, x_feat, z_feat_pos, x_feat_pos)

        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        outputs_class = outputs_class[-1]
        outputs_class = outputs_class.transpose(1, 2)

        return outputs_class, outputs_coord[-1]

