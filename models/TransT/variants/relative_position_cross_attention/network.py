import torch
import torch.nn as nn


def _get_single_scale(feat):
    if isinstance(feat, (list, tuple)):
        assert len(feat) == 1
        feat = feat[0]
    return feat


class RelativePositionTransTNetwork(nn.Module):
    def __init__(self, backbone, transformer, head):
        super(RelativePositionTransTNetwork, self).__init__()
        self.backbone = backbone
        self.transformer = transformer
        self.head = head

    def forward(self, z, x):
        z_feat = _get_single_scale(self.backbone(z))
        x_feat = _get_single_scale(self.backbone(x))

        feat = self.transformer(z_feat, x_feat)
        return self.head(feat)

    @torch.no_grad()
    def template(self, z):
        return _get_single_scale(self.backbone(z))

    @torch.no_grad()
    def track(self, z_feat, x):
        x_feat = _get_single_scale(self.backbone(x))

        feat = self.transformer(z_feat, x_feat)
        return self.head(feat)
