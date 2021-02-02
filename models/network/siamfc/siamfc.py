from torch import nn


class SiamFCNet(nn.Module):
    def __init__(self, backbone, head):
        super(SiamFCNet, self).__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, input_):
        z, x = input_
        z = self.backbone(z)
        x = self.backbone(x)
        return self.head(z, x)
