import torch.optim
import torch.optim
import math


def build_transt_optimizer(model, train_config):
    optimizer_config = train_config['optimization']['optimizer']
    if optimizer_config['type'] == 'SGD-SiamFC-v1':
        from .special.siamfc_layerwise import build_siamfc_layerwise_optimizer
        optimizer = build_siamfc_layerwise_optimizer(model, train_config)
    else:
        param_dicts = [
            {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": optimizer_config['lr_backbone']
            },
        ]

        lr = optimizer_config['lr']
        weight_decay = optimizer_config['weight_decay']

        if optimizer_config['type'] == 'SGD':
            momentum = optimizer_config['momentum']
            nesterov = optimizer_config['nesterov']
            optimizer = torch.optim.SGD(param_dicts, lr=lr, weight_decay=weight_decay, nesterov=nesterov, momentum=momentum)
        elif optimizer_config['type'] == 'AdamW':
            optimizer = torch.optim.AdamW(param_dicts, lr=lr,
                                          weight_decay=weight_decay)
        else:
            raise NotImplementedError(f'Unknown lr_scheduler {optimizer_config["type"]}')

    lr_scheduler_config = optimizer_config['lr_scheduler']

    if lr_scheduler_config['type'] == 'ExponentialLR':
        gamma = math.pow(
            lr_scheduler_config['ultimate_lr'] / optimizer_config['lr'],
            1.0 / train_config['optimization']['epochs'])
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)
    elif lr_scheduler_config['type'] == 'StepLR':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_scheduler_config['lr_drop'])
    else:
        raise NotImplementedError(f'Unknown lr_scheduler {lr_scheduler_config["type"]}')
    return optimizer, lr_scheduler
