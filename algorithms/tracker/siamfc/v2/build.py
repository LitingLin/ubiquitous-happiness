from models.backbone.alexnet import AlexNetV2
from models.head.siamfc import SiamFCBNHead
from models.network.siamfc import SiamFCNet


def build_net():
    backbone = AlexNetV2()
    head = SiamFCBNHead()
    network = SiamFCNet(backbone, head)
    return network
