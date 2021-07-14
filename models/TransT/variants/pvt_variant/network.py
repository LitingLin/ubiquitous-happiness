import torch
import torch.nn as nn


def _get_single_scale(feat):
    if isinstance(feat, (list, tuple)):
        assert len(feat) == 1
        feat = feat[0]
    return feat


class PVTFeatureFusionNetwork(nn.Module):
    def __init__(self, backbone, transformer, head,
                 template_output_stage, template_output_shape,
                 search_output_stage, search_output_shape):
        super(PVTFeatureFusionNetwork, self).__init__()
        self.backbone = backbone
        self.transformer = transformer
        self.head = head
        self.template_output_stage = template_output_stage
        self.template_output_shape = template_output_shape[1], template_output_shape[0]  # H, W
        self.search_output_stage = search_output_stage
        self.search_output_shape = search_output_shape[1], search_output_shape[0]  # H, W

    def _forward_feat(self, x, output_stage):
        return _get_single_scale(self.backbone(x, (output_stage,), False))

    def forward(self, z, x):
        z_feat = self._forward_feat(z, self.template_output_stage)
        x_feat = self._forward_feat(x, self.search_output_stage)

        feat = self.transformer(z_feat, x_feat, *self.template_output_shape, *self.search_output_shape)
        return self.head(feat.unsqueeze(0))

    @torch.no_grad()
    def template(self, z):
        return self._forward_feat(z, self.template_output_stage)

    @torch.no_grad()
    def track(self, z_feat, x):
        x_feat = self._forward_feat(x, self.search_output_stage)

        feat = self.transformer(z_feat, x_feat, *self.template_output_shape, *self.search_output_shape)
        return self.head(feat.unsqueeze(0))
