import torch
import torch.nn as nn


def _get_single_scale(feat):
    if isinstance(feat, (list, tuple)):
        assert len(feat) == 1
        feat = feat[0]
    return feat


class PVTFeatureFusionNetwork(nn.Module):
    def __init__(self, backbone, transformer, head,
                 transformer_hidden_dim, enable_input_projection,
                 template_output_stage, template_output_dim, template_output_shape,
                 search_output_stage, search_output_dim, search_output_shape):
        super(PVTFeatureFusionNetwork, self).__init__()
        self.backbone = backbone
        self.transformer = transformer
        self.head = head
        self.template_output_stage = template_output_stage
        self.template_output_shape = template_output_shape[1], template_output_shape[0]  # H, W
        self.search_output_stage = search_output_stage
        self.search_output_shape = search_output_shape[1], search_output_shape[0]  # H, W
        self.template_input_projection = None
        self.search_input_projection = None
        if enable_input_projection:
            self.template_input_projection = nn.Linear(template_output_dim, transformer_hidden_dim)
            self.search_input_projection = nn.Linear(search_output_dim, transformer_hidden_dim)

            nn.init.xavier_uniform_(self.template_input_projection.weight)
            nn.init.xavier_uniform_(self.search_input_projection.weight)

    def _forward_feat(self, x, output_stage, input_projection):
        x = _get_single_scale(self.backbone(x, (output_stage,), False))
        if input_projection is not None:
            x = input_projection(x)
        return x

    def forward(self, z, x):
        z_feat = self._forward_feat(z, self.template_output_stage, self.template_input_projection)
        x_feat = self._forward_feat(x, self.search_output_stage, self.search_input_projection)

        feat = self.transformer(z_feat, x_feat, *self.template_output_shape, *self.search_output_shape)
        return self.head(feat.unsqueeze(0))

    @torch.no_grad()
    def template(self, z):
        return self._forward_feat(z, self.template_output_stage, self.template_input_projection)

    @torch.no_grad()
    def track(self, z_feat, x):
        x_feat = self._forward_feat(x, self.search_output_stage, self.search_input_projection)

        feat = self.transformer(z_feat, x_feat, *self.template_output_shape, *self.search_output_shape)
        return self.head(feat.unsqueeze(0))
