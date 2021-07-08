import torch
import torch.nn as nn


def _get_single_scale(feat):
    if isinstance(feat, (list, tuple)):
        assert len(feat) == 1
        feat = feat[0]
    return feat


class BaselineTrackerNetwork(nn.Module):
    def __init__(self, backbone, feature_merger, transformer, head,
                 transformer_hidden_dim,
                 template_output_stage, template_output_dim,
                 search_output_stage, search_output_dim, search_output_shape):
        super(BaselineTrackerNetwork, self).__init__()
        self.backbone = backbone
        self.feature_merger = feature_merger
        self.transformer = transformer
        self.head = head
        self.template_output_stage = template_output_stage
        self.search_output_stage = search_output_stage

        self.template_input_projection = nn.Linear(template_output_dim, transformer_hidden_dim)
        self.search_input_projection = nn.Linear(search_output_dim, transformer_hidden_dim)

        self.query_embed = nn.Parameter(torch.empty((1, search_output_shape[0] * search_output_shape[1], transformer_hidden_dim), dtype=torch.float))
        nn.init.xavier_uniform_(self.template_input_projection.weight)
        nn.init.xavier_uniform_(self.search_input_projection.weight)
        nn.init.normal_(self.query_embed)

    def _forward_feat(self, x, output_stage, projection):
        x_feat = _get_single_scale(self.backbone(x, (output_stage, ), False))
        x_feat = projection(x_feat)
        return x_feat

    def forward(self, z, x):
        z_feat = self._forward_feat(z, self.template_output_stage, self.template_input_projection)
        x_feat = self._forward_feat(x, self.search_output_stage, self.search_input_projection)
        feat, pos = self.feature_merger(z_feat, x_feat)

        feat = self.transformer(feat, None, self.query_embed, pos)
        return self.head(feat)

    @torch.no_grad()
    def template(self, z):
        z_feat = self._forward_feat(z, self.template_output_stage, self.template_input_projection)
        return z_feat

    @torch.no_grad()
    def track(self, z_feat, x):
        x_feat = self._forward_feat(x, self.search_output_stage, self.search_input_projection)
        feat, pos = self.feature_merger(z_feat, x_feat)

        feat = self.transformer(feat, None, self.query_embed, pos)
        return self.head(feat)
