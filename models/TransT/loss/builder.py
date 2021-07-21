def build_criterion_and_weight_composer(network_config: dict, train_config: dict):
    head_type = network_config['head']['type']
    if head_type == 'GFocal-v2':
        from .gfocal import build_gfocal_loss
        criterion = build_gfocal_loss(network_config, train_config)
    else:
        raise NotImplementedError(head_type)

    from .loss_compose.builder import build_loss_composer
    loss_composer = build_loss_composer(train_config)
    return criterion, loss_composer
