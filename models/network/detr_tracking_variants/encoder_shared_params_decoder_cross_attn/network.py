import torch
from torch import nn

from models.modules.mlp import MLP


class DETRTracking(nn.Module):
    def __init__(self, backbone, transformer):
        super().__init__()
        hidden_dim = transformer.d_model
        self.backbone = backbone
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.transformer = transformer
        self.bbox_proj = MLP(hidden_dim, hidden_dim, 4, 3)

    def inference_template(self, z, z_mask):
        z_feat, z_feat_mask, z_feat_pos = self.backbone(z, z_mask)
        z_feat = self.input_proj(z_feat)
        return self.transformer.forward_template(z_feat, z_feat_mask, z_feat_pos)

    def inference_instance(self, z_encoded, z_mask, z_pos, x):
        n, _, h, w = x.shape
        x_mask = torch.zeros((n, h, w), device=x.device, dtype=torch.bool)
        x_feat, x_feat_mask, x_feat_pos = self.backbone(x, x_mask)
        x_feat = self.input_proj(x_feat)
        bbox_embed = self.transformer.forward_instance(z_encoded, z_mask, z_pos, x_feat, x_feat_mask, x_feat_pos)
        bbox = self.bbox_proj(bbox_embed[:, 0, :]).sigmoid()
        return bbox

    def forward(self, input_):
        z, z_mask, x, x_mask = input_
        z_feat, z_feat_mask, z_feat_pos = self.backbone(z, z_mask)
        x_feat, x_feat_mask, x_feat_pos = self.backbone(x, x_mask)

        z_feat = self.input_proj(z_feat)
        x_feat = self.input_proj(x_feat)

        bbox_embed = self.transformer(z_feat, x_feat, z_feat_mask, x_feat_mask, z_feat_pos, x_feat_pos)
        bbox = self.bbox_proj(bbox_embed[:, 0, :]).sigmoid()
        return bbox

    def reset_parameters(self):
        self.bbox_proj.reset_parameters()
        self.input_proj.reset_parameters()
