def build_transt_highway_network(network_config, load_pretrained=True):
    from .builders.backbone import build_backbone
    from .builders.position_encoding import build_position_encoding
    from .builders.feature_fusion import build_feature_fusion_network
    from .head.builder import build_highway_head

    backbone, cls_backbone, reg_backbone = build_backbone(network_config, load_pretrained)
    position_encoder, cls_position_encoder, reg_position_encoder = build_position_encoding(network_config)
    feature_fusion, cls_feature_fusion, reg_feature_fusion = build_feature_fusion_network(network_config)
    head = build_highway_head(network_config)

    from .network import TransTTraskHighwayTracking

    return TransTTraskHighwayTracking(backbone, cls_backbone, reg_backbone,
                                      position_encoder, cls_position_encoder, reg_position_encoder,
                                      feature_fusion, cls_feature_fusion, reg_feature_fusion,
                                      head)
