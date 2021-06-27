import copy


def build_logger(args, network_config, train_config):
    network_config = copy.deepcopy(network_config)
    assert 'train' not in network_config
    network_config['train'] = copy.deepcopy(train_config)
    assert 'running_vars' not in network_config
    network_config['running_vars'] = vars(args)

    from ._wandb import WandbLogger
    return WandbLogger('transt', ['test'], network_config, True)
