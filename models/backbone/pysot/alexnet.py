import torch
import torch.nn as nn
import os


class AlexNet(nn.Module):
    configs = [3, 96, 256, 384, 384, 256]

    def __init__(self, width_mult=1, output_layers=(4,)):
        configs = list(map(lambda x: 3 if x == 3 else
        int(x * width_mult), AlexNet.configs))
        super(AlexNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(configs[0], configs[1], kernel_size=11, stride=2),
            nn.BatchNorm2d(configs[1]),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(configs[1], configs[2], kernel_size=5),
            nn.BatchNorm2d(configs[2]),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(configs[2], configs[3], kernel_size=3),
            nn.BatchNorm2d(configs[3]),
            nn.ReLU(inplace=True),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(configs[3], configs[4], kernel_size=3),
            nn.BatchNorm2d(configs[4]),
            nn.ReLU(inplace=True),
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(configs[4], configs[5], kernel_size=3),
            nn.BatchNorm2d(configs[5]),
        )
        self.feature_size = configs[5]
        self.used_layers = output_layers

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)

        out = [x1, x2, x3, x4, x5]
        out = [out[i] for i in self.used_layers]
        if len(out) == 1:
            return out[0]
        else:
            return out

    def reset_parameters(self):
        for module in self.modules():
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()

    def load_pretrained(self):
        self.load_state_dict(
            torch.load(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'weight', 'pysot', 'alexnet-bn.pth'),
                       map_location='cpu'), strict=True)


def construct_alexnet(width_mult=1, output_layers=(4,)):
    net = AlexNet(width_mult=width_mult, output_layers=output_layers)
    net.num_channels_output = net.configs[1:]
    net.num_channels_output = [net.num_channels_output[i] for i in output_layers]
    return net
