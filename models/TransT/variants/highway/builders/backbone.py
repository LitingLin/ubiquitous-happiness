def build_backbone(network_config, load_pretrained=True):
    from .common import build_base_and_highway_networks
    import models.TransT._old.backbone

    return build_base_and_highway_networks(network_config, ('backbone', 'parameters'), ('backbone', 'highway'), ('classification', 'regression'), models.TransT._old.backbone.build_backbone, (load_pretrained,))
