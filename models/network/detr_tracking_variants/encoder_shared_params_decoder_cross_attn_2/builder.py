from ..encoder_shared_params_decoder_cross_attn.backbone import build_backbone
from models.transformer.detr_tracking_variants.encoder_shared_params_decoder_cross_attn.builder import build_transformer
from .network import DETRTracking
from ..encoder_shared_params_decoder_cross_attn.builder import initialize_detr_tracking_network


def build_detr_tracking_network(config: dict):
    backbone = build_backbone(config)
    transformer = build_transformer(config)
    return DETRTracking(backbone, transformer, config['transformer']['z_feats'])
