def build_position_encoding(network_config: dict):
    import models.TransT.position_encoding
    from .common import build_base_and_highway_networks

    return build_base_and_highway_networks(network_config, ('transformer',), ('transformer', 'highway'), ('classification', 'regression'), models.TransT.position_encoding.build_position_encoding, ())