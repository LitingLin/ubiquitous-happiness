def build_transt_variant_backbone_different_output_stage(network_config: dict, load_pretrained=True):
    from models.TransT.backbone import build_backbone
    assert network_config['backbone']['type'] == 'swin_transformer'
    backbone = build_backbone(network_config, load_pretrained)

    from .position_encoding import build_position_encoding

    position_encoder = build_position_encoding(network_config)

    from .feature_fusion import build_featurefusion_network
    feature_fusion_network = build_featurefusion_network(network_config)

    from models.TransT.head.builder import build_head

    head = build_head(network_config)

    transformer_hidden_dim = network_config['transformer']['hidden_dim']
    backbone_output_layers_config = network_config['transformer']['backbone_output_layers']
    template_output_stage = backbone_output_layers_config['template']['stage']
    template_output_shape = backbone_output_layers_config['template']['shape']
    template_output_dim = backbone_output_layers_config['template']['dim']

    search_output_stage = backbone_output_layers_config['search']['stage']
    search_output_shape = backbone_output_layers_config['search']['shape']
    search_output_dim = backbone_output_layers_config['search']['dim']

    from .network import TransTVariantBackboneDifferentOutputStageNetwork

    return TransTVariantBackboneDifferentOutputStageNetwork(backbone, position_encoder, feature_fusion_network, head,
                                                            transformer_hidden_dim,
                                                            template_output_stage, template_output_dim, template_output_shape,
                                                            search_output_stage, search_output_dim, search_output_shape)