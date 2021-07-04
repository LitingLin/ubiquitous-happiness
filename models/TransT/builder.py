def build_transt(network_config, load_pretrained=True):
    if 'version' not in network_config:
        from models.TransT._old.builder import build_transt_old
        return build_transt_old(network_config, load_pretrained)

    if network_config['version'] == 2:
        from models.TransT.feature_fusion.feature_fusion import build_featurefusion_network
        from models.TransT._old.old_backbone import build_backbone
        from models.TransT._old.network import TransTTracking
        from models.TransT.head._old.builder import build_head

        transformer = build_featurefusion_network(network_config)
        backbone = build_backbone(network_config, load_pretrained)
        head = build_head(network_config)

        return TransTTracking(backbone, transformer, head)
    elif network_config['version'] == 3 and network_config['type'] == 'XTracker':
        from .variants.swin_cross_tracker import build_swin_transformer_x_tracker
        return build_swin_transformer_x_tracker(network_config, load_pretrained)
    elif network_config['version'] == 4 and network_config['type'].startswith('SiamFC'):
        from .siamfc.builder import build_siamfc
        return build_siamfc(network_config, load_pretrained)
    elif network_config['version'] == 4 and network_config['type'] == 'TransT':
        from models.TransT.feature_fusion.feature_fusion import build_featurefusion_network
        from models.TransT._old.backbone import build_backbone
        from models.TransT._old.network import TransTTracking
        from models.TransT.head.builder import build_head

        transformer = build_featurefusion_network(network_config)
        backbone = build_backbone(network_config, load_pretrained)
        head = build_head(network_config)

        return TransTTracking(backbone, transformer, head)
    elif network_config['version'] == 5 and network_config['type'] == 'TransT':
        from models.TransT.feature_fusion.feature_fusion import build_featurefusion_network
        from models.TransT.backbone import build_backbone
        from models.TransT.network import TransTTracking
        from models.TransT.head.builder import build_head
        from models.TransT.position_encoding import build_position_encoding

        transformer = build_featurefusion_network(network_config)
        backbone = build_backbone(network_config, load_pretrained)
        position_encoding = build_position_encoding(network_config)
        head = build_head(network_config)

        return TransTTracking(backbone, position_encoding, transformer, head)
    elif network_config['version'] == 5 and network_config['type'] == 'TransT-Task-Highway':
        from .variants.highway.builder import build_transt_highway_network
        return build_transt_highway_network(network_config, load_pretrained)
    else:
        raise NotImplementedError(f'Unknown version {network_config["version"]}')
