from models.backbone.alexnet import AlexNetV1
from models.head.siamfc import SiamFCLinearHead
from models.network.siamfc import SiamFCNet


def build_net():
    backbone = AlexNetV1()
    head = SiamFCLinearHead()
    network = SiamFCNet(backbone, head)
    return network
