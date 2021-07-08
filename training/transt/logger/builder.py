import copy
from datetime import datetime
from miscellanies.torch.distributed import get_world_size


def _get_wandb_config(network_config: dict):
    if 'logging' in network_config:
        return network_config['logging']['category'], network_config['logging']['tags']
    else:
        if 'category' in network_config:
            category = network_config['category']
        else:
            category = 'transt'
        if 'tags' in network_config:
            tags = network_config['tags']
        else:
            tags = None
        return category, tags


def build_logger(args, network_config, train_config, initial_step):
    config_name = args.config_name
    logger_id = f'{config_name}-{datetime.now().strftime("%Y.%m.%d-%H.%M.%S-%f")}'
    network_config = copy.deepcopy(network_config)
    assert 'train' not in network_config
    network_config['train'] = copy.deepcopy(train_config)
    assert 'running_vars' not in network_config
    network_config['running_vars'] = vars(args)
    tensorboard_root_path = None

    if args.disable_wandb:
        from .dummy import DummyLogger
        return DummyLogger()

    from ._wandb import WandbLogger, has_wandb
    if has_wandb:
        wandb_project, wandb_tags = _get_wandb_config(network_config)
        return WandbLogger(logger_id, wandb_project, network_config,
                           wandb_tags, get_world_size(),
                           initial_step, args.logging_interval,
                           True, args.watch_model_freq,
                           args.watch_model_parameters, args.watch_model_gradients,
                           tensorboard_root_path)
    else:
        from .dummy import DummyLogger
        return DummyLogger()
