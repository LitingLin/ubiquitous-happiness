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
        return x_feat, self.position_encoding(x_feat)


def build_backbone(net_config: dict):
    backbone_config = net_config['backbone']
    position_encoding = build_position_encoding(net_config)
    backbone_build_params = {}
    if 'output_layers' in backbone_config:
        backbone_build_params['output_layers'] = backbone_config['output_layers']
    if 'dilation' in backbone_config:
        backbone_build_params['dilation'] = backbone_config['dilation']
    if backbone_config['type'] == 'alexnet':
        from models.backbone.pysot.alexnet import construct_alexnet
        backbone = construct_alexnet(**backbone_build_params)
    elif backbone_config['type'] == 'resnet50_atrous':
        from models.backbone.pysot.resnet_atrous import construct_resnet50_atrous
        backbone = construct_resnet50_atrous(**backbone_build_params)
    elif backbone_config['type'] == 'resnet50_detr':
        from models.backbone.detr_tracking.resnet import construct_resnet50
        backbone = construct_resnet50(**backbone_build_params)
    else:
        raise Exception(f'unsupported {backbone_config["type"]}')

    return TransTBackbone(backbone, position_encoding)
