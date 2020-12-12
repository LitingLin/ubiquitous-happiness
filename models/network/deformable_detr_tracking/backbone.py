from models.transformer.position_encoding import build_position_encoding
from torch import nn


class DeformDETRTrackBackbonePipeline(nn.Module):
    def __init__(self, backbone, position_encoding):
        self.backbone = backbone
        self.position_encoding = position_encoding

    def forward(self, x, x_mask, z, z_mask):




def build_backbone(net_config: dict):
    backbone_config = net_config['backbone']
    position_encoding = build_position_encoding(net_config)
    if backbone_config['type'] == 'alexnet':
        from models.backbone.pysot.alexnet import construct_alexnet
        backbone = construct_alexnet(output_layers=backbone_config['output_layers'])
    elif backbone_config['type'] == 'resnet50':
        from models.backbone.pysot.resnet_atrous import construct_resnet50_atrous
        backbone = construct_resnet50_atrous(output_layers=backbone_config['output_layers'])
    else:
        raise Exception(f'unsupported {backbone_config["type"]}')

    return backbone, position_encoding