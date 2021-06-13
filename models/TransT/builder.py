def build_transt(network_config, load_pretrained=True):
    if 'version' not in network_config:
        from models.TransT._old.builder import build_transt_old
        return build_transt_old(network_config, load_pretrained)

    if network_config['version'] == 2:
        from .feature_fusion import build_featurefusion_network
        from .backbone import build_backbone
        from .network import TransTTracking
        from .head.builder import build_head

        transformer = build_featurefusion_network(network_config)
        backbone = build_backbone(network_config, load_pretrained)
        head = build_head(network_config)

        return TransTTracking(backbone, transformer, head)
    elif network_config['version'] == 3 and network_config['type'] == 'XTracker':
        from .variants.swin_cross_tracker import build_swin_transformer_x_tracker
        return build_swin_transformer_x_tracker(network_config, load_pretrained)
    else:
        raise NotImplementedError(f'Unknown version {network_config["version"]}')
