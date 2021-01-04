from models.transformer.position_encoding import build_position_encoding
from models.transformer.detr_siam.transformer import build_transformer
from .backbone import SiamEncoderBackboneMaskWrapper
from .network import DETR


# TODO: multi level feat, construct with output_layers=()
def _build_backbone(config: dict):
    backbone_config = config['backbone']
    position_encoding = build_position_encoding(config)
    if backbone_config['type'] == 'alexnet':
        from models.backbone.pysot.alexnet import construct_alexnet
        backbone = construct_alexnet()
    elif backbone_config['type'] == 'resnet50':
        from models.backbone.pysot.resnet_atrous import construct_resnet50_atrous
        backbone = construct_resnet50_atrous()
    else:
        raise Exception(f'unsupported {backbone_config["type"]}')

    return backbone, position_encoding


def initialize_siam_encoder_detr_track(model, backbone_load_pretrained=True):
    model.reset_parameters()
    backbone_wrapper = model.backbone
    backbone = backbone_wrapper.backbone
    position_encoding = backbone_wrapper.position_encoding
    if hasattr(position_encoding, 'reset_parameters'):
        position_encoding.reset_parameters()

    if backbone_load_pretrained:
        backbone.load_pretrained()
    else:
        backbone.reset_parameters()
    transformer = model.transformer
    transformer.reset_parameters()


def build_siam_encoder_detr_track(config: dict):
    backbone, position_encoding = _build_backbone(config)
    transformer = build_transformer(config)
    backbone_wrapper = SiamEncoderBackboneMaskWrapper(backbone, position_encoding)
    return DETR(backbone_wrapper, transformer, config['transformer']['num_queries'])
