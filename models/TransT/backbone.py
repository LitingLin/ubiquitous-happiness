from .position_encoding import build_position_encoding
from torch import nn


class TransTBackbone(nn.Module):
    def __init__(self, backbone, position_encoding):
        super(TransTBackbone, self).__init__()
        self.backbone = backbone
        self.position_encoding = position_encoding
        assert len(self.backbone.num_channels_output) == 1
        self.num_channels = self.backbone.num_channels_output[0]

    def forward(self, x):
        x_feat = self.backbone(x)
        if isinstance(x_feat, (list, tuple)):
            x_feat = x_feat[0]
        return x_feat, self.position_encoding(x_feat)


def build_backbone(net_config: dict, load_pretrained=True):
    backbone_config = net_config['backbone']
    position_encoding = build_position_encoding(net_config)
    if 'parameters' in backbone_config:
        backbone_build_params = backbone_config['parameters']
    else:
        backbone_build_params = ()
    if backbone_config['type'] == 'alexnet':
        from models.backbone.pysot.alexnet import construct_alexnet
        backbone = construct_alexnet(load_pretrained, **backbone_build_params)
    elif backbone_config['type'] == 'resnet50_atrous':
        from models.backbone.pysot.resnet_atrous import construct_resnet50_atrous
        backbone = construct_resnet50_atrous(load_pretrained, **backbone_build_params)
    elif backbone_config['type'] == 'resnet50_detr':
        from models.backbone.detr_tracking.resnet import construct_resnet50
        backbone = construct_resnet50(load_pretrained, **backbone_build_params)
    elif backbone_config['type'] == 'swin_transformer':
        from models.backbone.swint.swin_transformer import build_swint_backbone
        backbone = build_swint_backbone(load_pretrained=load_pretrained, **backbone_build_params)
    else:
        raise Exception(f'unsupported {backbone_config["type"]}')

    return TransTBackbone(backbone, position_encoding)
