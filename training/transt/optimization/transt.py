import torch.optim
import math


def build_transt_optimizer(model, train_config):
    from .optimizer_builder.builder import build_optimizer
    optimizer = build_optimizer(model, train_config)
    optimizer_config = train_config['optimization']['optimizer']
    lr_scheduler_config = optimizer_config['lr_scheduler']

    if lr_scheduler_config['type'] == 'ExponentialLR':
        if 'ultimate_lr' in lr_scheduler_config:
            gamma = math.pow(
                lr_scheduler_config['ultimate_lr'] / optimizer_config['lr'],
                1.0 / train_config['optimization']['epochs'])
        else:
            gamma = lr_scheduler_config['gamma']
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)
    elif lr_scheduler_config['type'] == 'StepLR':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_scheduler_config['lr_drop'])
    else:
        raise NotImplementedError(f'Unknown lr_scheduler {lr_scheduler_config["type"]}')
    return optimizer, lr_scheduler
