from .alexnet import AlexNet, AlexNetLegacy
from .mobile_v2 import MobileNetV2
from .resnet_atrous import resnet18, resnet34, resnet50


_BACKBONES = {
              'alexnetlegacy': AlexNetLegacy,
              'mobilenetv2': MobileNetV2,
              'resnet18': resnet18,
              'resnet34': resnet34,
              'resnet50': resnet50,
              'alexnet': AlexNet,
            }


def build_backbone(config: dict):
    return _BACKBONES[config['BACKBONE']['TYPE']](**config['BACKBONE']['KWARGS'])
