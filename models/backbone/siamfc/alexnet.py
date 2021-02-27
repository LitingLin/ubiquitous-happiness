# https://github.com/huanglianghua/siamfc-pytorch/blob/master/siamfc/backbones.py
import torch.nn as nn

__all__ = ['AlexNetV1', 'AlexNetV2', 'AlexNetV3']


class _AlexNet(nn.Module):
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x


class AlexNetV1(_AlexNet):
    output_stride = 8

    def __init__(self):
        super(AlexNetV1, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 96, 11, 2),
            nn.BatchNorm2d(96, eps=1e-5, momentum=0.05),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, 5, 1, groups=2),
            nn.BatchNorm2d(256, eps=1e-5, momentum=0.05),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, 3, 1),
            nn.BatchNorm2d(384, eps=1e-5, momentum=0.05),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 384, 3, 1, groups=2),
            nn.BatchNorm2d(384, eps=1e-5, momentum=0.05),
            nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(
            nn.Conv2d(384, 256, 3, 1, groups=2))


class AlexNetV2(_AlexNet):
    output_stride = 4

    def __init__(self):
        super(AlexNetV2, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 96, 11, 2),
            nn.BatchNorm2d(96, eps=1e-5, momentum=0.05),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, 5, 1, groups=2),
            nn.BatchNorm2d(256, eps=1e-5, momentum=0.05),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 1))
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, 3, 1),
            nn.BatchNorm2d(384, eps=1e-5, momentum=0.05),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 384, 3, 1, groups=2),
            nn.BatchNorm2d(384, eps=1e-5, momentum=0.05),
            nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(
            nn.Conv2d(384, 32, 3, 1, groups=2))


class AlexNetV3(_AlexNet):
    output_stride = 8

    def __init__(self):
        super(AlexNetV3, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 192, 11, 2),
            nn.BatchNorm2d(192, eps=1e-5, momentum=0.05),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(192, 512, 5, 1),
            nn.BatchNorm2d(512, eps=1e-5, momentum=0.05),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(512, 768, 3, 1),
            nn.BatchNorm2d(768, eps=1e-5, momentum=0.05),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(768, 768, 3, 1),
            nn.BatchNorm2d(768, eps=1e-5, momentum=0.05),
            nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(
            nn.Conv2d(768, 512, 3, 1),
            nn.BatchNorm2d(512, eps=1e-5, momentum=0.05))
