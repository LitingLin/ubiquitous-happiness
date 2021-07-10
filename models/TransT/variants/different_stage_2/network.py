import torch
import torch.nn as nn
from models.TransT.operator.sine_position_encoding import generate_2d_sine_position_encoding


def _get_single_scale(feat):
    if isinstance(feat, (list, tuple)):
        assert len(feat) == 1
        feat = feat[0]
    return feat


class TransTVariantBackboneDifferentOutputStageNetwork(nn.Module):
    def __init__(self, backbone, transformer, head,
                 transformer_hidden_dim,
                 template_output_stage, template_output_dim, template_output_shape,
                 search_output_stage, search_output_dim, search_output_shape):
        super(TransTVariantBackboneDifferentOutputStageNetwork, self).__init__()
        self.backbone = backbone
        self.transformer = transformer
        self.head = head
        self.template_output_stage = template_output_stage
        self.search_output_stage = search_output_stage

        self.template_input_projection = nn.Linear(template_output_dim, transformer_hidden_dim)
        self.search_input_projection = nn.Linear(search_output_dim, transformer_hidden_dim)

        nn.init.xavier_uniform_(self.template_input_projection.weight)
        nn.init.xavier_uniform_(self.search_input_projection.weight)

        self.template_output_shape = template_output_shape
        self.search_output_shape = search_output_shape

        self.register_buffer('template_position_encoding',
                             generate_2d_sine_position_encoding(1, template_output_shape[1], template_output_shape[0], transformer_hidden_dim).view(1, template_output_shape[1] * template_output_shape[0], transformer_hidden_dim))

        self.register_buffer('search_position_encoding',
                             generate_2d_sine_position_encoding(1, search_output_shape[1], search_output_shape[0], transformer_hidden_dim).view(1, search_output_shape[1] * search_output_shape[0], transformer_hidden_dim))

    def _forward_feat(self, x, output_stage, projection):
        x_feat = _get_single_scale(self.backbone(x, (output_stage, ), False))
        x_feat = projection(x_feat)
        return x_feat

    def forward(self, z, x):
        z_feat = self._forward_feat(z, self.template_output_stage, self.template_input_projection)
        x_feat = self._forward_feat(x, self.search_output_stage, self.search_input_projection)

        feat = self.transformer(z_feat, x_feat, self.template_position_encoding, self.search_position_encoding)
        return self.head(feat)

    @torch.no_grad()
    def template(self, z):
        z_feat = self._forward_feat(z, self.template_output_stage, self.template_input_projection)
        return z_feat

    @torch.no_grad()
    def track(self, z, x):
        z_feat = z
        x_feat = self._forward_feat(x, self.search_output_stage, self.search_input_projection)

        feat = self.transformer(z_feat, x_feat, self.template_position_encoding, self.search_position_encoding)
        return self.head(feat)
