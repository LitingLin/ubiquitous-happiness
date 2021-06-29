import copy
from datetime import datetime


def build_logger(args, network_config, train_config, initial_step):
    config_name = args.config_name
    logger_id = f'{config_name}-{datetime.now().strftime("%Y.%m.%d-%H.%M.%S-%f")}'
    network_config = copy.deepcopy(network_config)
    assert 'train' not in network_config
    network_config['train'] = copy.deepcopy(train_config)
    assert 'running_vars' not in network_config
    network_config['running_vars'] = vars(args)
    tensorboard_root_path = None

    from ._wandb import WandbLogger, has_wandb
    if has_wandb:
        return WandbLogger(logger_id, 'transt', network_config, initial_step, args.logging_interval,
                           True, args.watch_model_freq,
                           args.watch_model_parameters, args.watch_model_gradients,
                           tensorboard_root_path)
    else:
        from .dummy import DummyLogger
        return DummyLogger()
