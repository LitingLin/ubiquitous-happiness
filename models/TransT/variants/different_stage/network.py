import torch
import torch.nn as nn


def _get_single_scale(feat):
    if isinstance(feat, (list, tuple)):
        assert len(feat) == 1
        feat = feat[0]
    return feat


class TransTVariantBackboneDifferentOutputStageNetwork(nn.Module):
    def __init__(self, backbone, position_encoder, transformer, head,
                 transformer_hidden_dim,
                 template_output_stage, template_output_dim, template_output_shape,
                 search_output_stage, search_output_dim, search_output_shape):
        super(TransTVariantBackboneDifferentOutputStageNetwork, self).__init__()
        self.backbone = backbone
        self.position_encoder = position_encoder
        self.transformer = transformer
        self.head = head
        self.template_output_stage = template_output_stage
        self.search_output_stage = search_output_stage

        self.template_input_projection = nn.Linear(template_output_dim, transformer_hidden_dim)
        self.search_input_projection = nn.Linear(search_output_dim, transformer_hidden_dim)

        self.template_output_shape = template_output_shape
        self.search_output_shape = search_output_shape

    def _forward_feat(self, x, output_stage, output_shape, projection):
        x_feat = _get_single_scale(self.backbone(x, output_stage, False))
        x_feat = projection(x_feat)
        x_pos = self.position_encoder(x_feat, *output_shape)
        return x_feat, x_pos

    def forward(self, z, x):
        z_feat, z_pos = self._forward_feat(z, self.template_output_stage, self.template_output_shape, self.template_input_projection)
        x_feat, x_pos = self._forward_feat(x, self.search_output_stage, self.search_output_shape, self.search_input_projection)

        feat = self.transformer(z_feat, x_feat, z_pos, x_pos)
        return self.head(feat)

    @torch.no_grad()
    def template(self, z):
        z_feat, z_pos = self._forward_feat(z, self.template_output_stage, self.template_output_shape,
                                           self.template_input_projection)
        return z_feat, z_pos

    @torch.no_grad()
    def track(self, z, x):
        z_feat, z_pos = z
        x_feat, x_pos = self._forward_feat(x, self.search_output_stage, self.search_output_shape, self.search_input_projection)

        feat = self.transformer(z_feat, x_feat, z_pos, x_pos)
        return self.head(feat)
