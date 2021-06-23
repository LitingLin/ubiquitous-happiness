import torch
import torch.nn as nn


class SiamFCNetwork(nn.Module):
    def __init__(self, backbone, neck, head):
        super(SiamFCNetwork, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head

    def forward(self, z, x):
        z = self.backbone(z)
        x = self.backbone(x)
        cross = self.neck(z, x)
        return self.head(cross)

    @torch.no_grad()
    def template(self, z):
        return self.backbone(z)

    @torch.no_grad()
    def track(self, z_feat, x):
        x = self.backbone(x)
        cross = self.neck(z_feat, x)
        return self.head(cross)


class SiamFCDualPathNetwork(nn.Module):
    def __init__(self, backbone, neck, head):
        super(SiamFCDualPathNetwork, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head

    def forward(self, z, x):
        z = self.backbone(z)
        x = self.backbone(x)
        cls, reg = self.neck(z, x)
        return self.head(cls, reg)

    @torch.no_grad()
    def template(self, z):
        return self.backbone(z)

    @torch.no_grad()
    def track(self, z_feat, x):
        x = self.backbone(x)
        cls, reg = self.neck(z_feat, x)
        return self.head(cls, reg)

