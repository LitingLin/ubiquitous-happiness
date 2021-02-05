from .backbone import build_backbone
from models.transformer.detr_tracking_variants.encoder_shared_params_decoder_cross_attn.builder import build_transformer
from .network import DETRTracking


def build_detr_tracking_network(config: dict):
    backbone = build_backbone(config)
    transformer = build_transformer(config)
    return DETRTracking(backbone, transformer)


def initialize_detr_tracking_network(model, backbone_load_pretrained=True):
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
