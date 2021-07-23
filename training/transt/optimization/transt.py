import torch.optim
import math


def build_transt_optimizer(model, train_config, iterations_per_epoch: int):
    from .optimizer_builder.builder import build_optimizer
    optimizer = build_optimizer(model, train_config)
    optimizer_config = train_config['optimization']['optimizer']
    lr_scheduler_config = optimizer_config['lr_scheduler']

    if 'per_iteration' in lr_scheduler_config:
        from training.solver.lr_scheduler.builder import build_iter_wise_lr_scheduler
        lr_scheduler = build_iter_wise_lr_scheduler(optimizer, train_config, iterations_per_epoch)
        per_iteration = True
    else:
        per_iteration = False
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
    return optimizer, lr_scheduler, per_iteration
