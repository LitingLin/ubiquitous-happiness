def build_criterion_and_weight_composer(network_config: dict, train_config: dict, iterations_per_epoch):
    head_type = network_config['head']['type']
    if head_type == 'GFocal-v2':
        from .gfocal import build_gfocal_loss
        criterion = build_gfocal_loss(network_config, train_config)
    elif head_type == 'TransT':
        from .transt import build_transt_loss
        criterion = build_transt_loss(train_config)
    else:
        raise NotImplementedError(head_type)

    from .loss_compose.builder import build_loss_composer
    loss_composer = build_loss_composer(train_config, iterations_per_epoch)
    return criterion, loss_composer
