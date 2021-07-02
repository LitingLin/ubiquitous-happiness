import torch.optim


def build_optimizer(optimizer_params: list, train_config: dict):
    optimizer_config = train_config['optimization']['optimizer']
    lr = optimizer_config['lr']
    weight_decay = optimizer_config['weight_decay']

    if optimizer_config['type'] == 'SGD':
        momentum = optimizer_config['momentum']
        nesterov = optimizer_config['nesterov']
        optimizer = torch.optim.SGD(optimizer_params, lr=lr, weight_decay=weight_decay, nesterov=nesterov, momentum=momentum)
    elif optimizer_config['type'] == 'AdamW':
        optimizer = torch.optim.AdamW(optimizer_params, lr=lr,
                                      weight_decay=weight_decay)
    else:
        raise NotImplementedError(f'Unknown lr_scheduler {optimizer_config["type"]}')

    return optimizer


def build_common_optimizer(model, train_config: dict):
    return build_optimizer(model.parameters(), train_config)
