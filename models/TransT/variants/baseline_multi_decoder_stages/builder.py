def build_baseline_tracker(network_config: dict, load_pretrained=True):
    from models.TransT.backbone import build_backbone
    assert network_config['backbone']['type'] == 'swin_transformer'
    backbone = build_backbone(network_config, load_pretrained)

    from .feature_merging import build_feature_merging

    feature_merging = build_feature_merging(network_config)

    from .transformer import build_transformer
    feature_fusion_network = build_transformer(network_config)

    from models.TransT.head.builder import build_head

    head = build_head(network_config)

    transformer_hidden_dim = network_config['transformer']['hidden_dim']
    transformer_num_queries = network_config['transformer']['num_queries']
    backbone_output_layers_config = network_config['transformer']['backbone_output_layers']
    template_output_stage = backbone_output_layers_config['template']['stage']
    template_output_shape = backbone_output_layers_config['template']['shape']
    template_output_dim = backbone_output_layers_config['template']['dim']

    search_output_stage = backbone_output_layers_config['search']['stage']
    search_output_shape = backbone_output_layers_config['search']['shape']
    search_output_dim = backbone_output_layers_config['search']['dim']

    from .network import BaselineTrackerNetwork

    return BaselineTrackerNetwork(backbone, feature_merging, feature_fusion_network, head,
                                  transformer_num_queries,
                                  transformer_hidden_dim,
                                  template_output_stage, template_output_dim,
                                  search_output_stage, search_output_dim)
