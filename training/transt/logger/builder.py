import copy
from datetime import datetime


def build_logger(args, network_config, train_config):
    config_name = args.config_name
    logger_id = f'{config_name}-{datetime.now().strftime("%Y.%m.%d-%H.%M.%S")}'
    network_config = copy.deepcopy(network_config)
    assert 'train' not in network_config
    network_config['train'] = copy.deepcopy(train_config)
    assert 'running_vars' not in network_config
    network_config['running_vars'] = vars(args)

    from ._wandb import WandbLogger
    return WandbLogger(logger_id, 'transt', network_config, True)
