def build_transt_old(network_config: dict, load_pretrained=True):
    from models.TransT.feature_fusion import build_featurefusion_network
    from models.TransT._old.old_backbone import build_backbone
    from .network_v1 import TransTTracking
    transformer = build_featurefusion_network(network_config)
    backbone = build_backbone(network_config, load_pretrained)
    return TransTTracking(backbone, transformer)
