from models.TransT.backbone import build_backbone
from .feature_fusion import build_feature_fusion_network
from models.TransT.head.builder import build_head
from .network import RelativePositionTransTNetwork


def build_relative_position_transt_tracker(network_config, load_pretrained=True):
    backbone = build_backbone(network_config, load_pretrained)
    transformer = build_feature_fusion_network(network_config)
    head = build_head(network_config)

    return RelativePositionTransTNetwork(backbone, transformer, head)
