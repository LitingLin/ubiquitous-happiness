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
    elif network_config['version'] == 5 and network_config['type'] == 'TransT-SwinT-CrossAttention':
        from .variants.relative_position_cross_attention.builder import build_relative_position_transt_tracker
        return build_relative_position_transt_tracker(network_config, load_pretrained)
    elif network_config['version'] == 5 and network_config['type'] == 'TransT-AFT':
        raise NotImplementedError('AFT was rejected by OpenReview')
        from .variants.aft_feature_fusion.builder import build_aft_transt_tracker
        return build_aft_transt_tracker(network_config, load_pretrained)
    elif network_config['version'] == 5 and network_config['type'] == 'TransT-Different-Output-Stage':
        from .variants.different_stage.builder import build_transt_variant_backbone_different_output_stage
        return build_transt_variant_backbone_different_output_stage(network_config, load_pretrained)
    elif network_config['version'] == 5 and network_config['type'] == 'TransT-Different-Output-Stage-2':
        from .variants.different_stage_2.builder import build_transt_variant_backbone_different_output_stage
        return build_transt_variant_backbone_different_output_stage(network_config, load_pretrained)
    elif network_config['version'] == 5 and network_config['type'] == 'T-Baseline':
        from .variants.baseline.builder import build_baseline_tracker
        return build_baseline_tracker(network_config, load_pretrained)
    elif network_config['version'] == 5 and network_config['type'] == 'T-Baseline-multi-decoder-stage':
        from .variants.baseline_multi_decoder_stages.builder import build_baseline_tracker
        return build_baseline_tracker(network_config, load_pretrained)
    elif network_config['version'] == 5 and network_config['type'] == 'TransT-SwinT-CrossAttention-lepe':
        raise NotImplementedError('Consuming too much memory')
        from .variants.lepe_cross_attention.builder import build_relative_position_transt_tracker
        return build_relative_position_transt_tracker(network_config, load_pretrained)
    elif network_config['version'] == 5 and network_config['type'] == 'TransT-PVT':
        from .variants.pvt_variant.builder import build_pvt_feature_fusion_tracker
        return build_pvt_feature_fusion_tracker(network_config, load_pretrained)
    else:
        raise NotImplementedError(f'Unknown version {network_config["version"]}')
