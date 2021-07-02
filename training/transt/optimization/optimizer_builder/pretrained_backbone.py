def build_pretrained_backbone_optimzer(model, train_config: dict):
    optimizer_config = train_config['optimization']['optimizer']
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": optimizer_config['advanced_strategy']['lr_backbone']
        },
    ]
    from .common import build_optimizer
    return build_optimizer(param_dicts, train_config)
