def build_criterion(train_config: dict):
    if 'version' not in train_config:
        from ._old.builder import build_transt_criterion_old
        return build_transt_criterion_old(train_config)

    assert train_config['version'] == 2

    loss_type = train_config['train']['loss']['type']

    if loss_type == 'transt':
        from .transt import build_transt_criterion
        return build_transt_criterion(train_config)
    elif loss_type == 'exp-1':
        from .exp_1 import build_transt_criterion
        return build_transt_criterion(train_config)
    else:
        raise RuntimeError(f"Unknown value {loss_type}")
