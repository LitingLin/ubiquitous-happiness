from torch import nn


class SiamFCMultiResNet(nn.Module):
    def __init__(self, backbone, heads):
        super(SiamFCMultiResNet, self).__init__()
        self.backbone = backbone
        if isinstance(heads, list):
            heads = nn.ModuleList(heads)
        self.heads = heads
        self.num_channels_output = backbone.num_channels_output

    def forward(self, z, x):
        z = self.backbone(z)
        x = self.backbone(x)
        output = []
        for i, (i_z, i_x) in enumerate(zip(z, x)):
            output.append(self.heads[i](i_z, i_x))
        return output
