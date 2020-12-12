from torch import nn


class SiamFCNet(nn.Module):
    def __init__(self, backbone, head):
        super(SiamFCNet, self).__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, z, x):
        z = self.backbone(z)
        x = self.backbone(x)
        return self.head(z, x)
