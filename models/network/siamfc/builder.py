import models.backbone.siamfc.alexnet as backbone
import models.head.siamfc.siamfc as head
from models.network.siamfc.siamfc import SiamFCNet


def build_siamfc_network(config: dict):
    model_config = config['model']
    backbone_type = model_config['backbone']['type']
    backbone_version = model_config['backbone']['version']
    if backbone_type == 'alexnet' and backbone_version == 1:
        model_backbone = backbone.AlexNetV1()
    elif backbone_type == 'alexnet' and backbone_version == 2:
        model_backbone = backbone.AlexNetV2()
    else:
        raise NotImplementedError

    head_type = model_config['head']['type']
    if head_type == 'linear':
        model_head = head.SiamFCLinearHead()
    elif head_type == 'batch_norm':
        model_head = head.SiamFCBNHead()
    else:
        raise NotImplementedError
    return SiamFCNet(model_backbone, model_head)
