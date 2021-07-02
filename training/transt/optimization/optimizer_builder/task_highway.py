from models.TransT.variants.highway.network import TransTTraskHighwayTracking


def build_task_highway_optimizer(model: TransTTraskHighwayTracking, train_config: dict):
    optimizer_config = train_config['optimization']['optimizer']
    lr = optimizer_config['lr']
    lr_backbone = optimizer_config['advanced_strategy']['lr_backbone']

    lr_classification_branch = optimizer_config['advanced_strategy']['lr_classification_ratio'] * lr
    lr_regression_branch = optimizer_config['advanced_strategy']['lr_regression_ratio'] * lr

    classification_branch_params = []
    regression_branch_params = []
    backbone_params = []
    rest_params = []

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if 'classification' in n:
            classification_branch_params.append(p)
        elif 'regression' in n:
            regression_branch_params.append(p)
        elif 'backbone' in n:
            backbone_params.append(p)
        else:
            rest_params.append(p)

    optimizer_params = [
        {'params': rest_params},
        {'params': backbone_params, 'lr': lr_backbone},
        {'params': classification_branch_params, 'lr': lr_classification_branch},
        {'params': regression_branch_params, 'lr': lr_regression_branch},
    ]

    from .common import build_optimizer
    return build_optimizer(optimizer_params, train_config)
