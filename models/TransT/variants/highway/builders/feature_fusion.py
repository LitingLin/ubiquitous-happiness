def build_feature_fusion_network(network_config: dict):
    import models.TransT.feature_fusion.feature_fusion
    from .common import build_base_and_highway_networks

    return build_base_and_highway_networks(network_config, ('transformer',), ('transformer', 'highway'), ('classification', 'regression'), models.TransT.feature_fusion.feature_fusion.build_featurefusion_network, ())
