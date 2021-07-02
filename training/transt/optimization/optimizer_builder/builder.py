def build_optimizer(model, train_config: dict):
    optimizer_config = train_config['optimization']['optimizer']
    if 'advanced_strategy' not in optimizer_config:
        from .common import build_common_optimizer
        return build_common_optimizer(model, train_config)
    else:
        advanced_strategy = optimizer_config['advanced_strategy']
        if advanced_strategy['type'] == 'SiamFC':
            from .siamfc_layerwise import build_siamfc_layerwise_optimizer
            return build_siamfc_layerwise_optimizer(model, train_config)
        elif advanced_strategy['type'] == 'Backbone-Pretrained':
            from .pretrained_backbone import build_pretrained_backbone_optimzer
            return build_pretrained_backbone_optimzer(model, train_config)
        elif advanced_strategy['type'] == 'Task-Highway':
            from .task_highway import build_task_highway_optimizer
            return build_task_highway_optimizer(model, train_config)
        else:
            raise NotImplementedError(f'Unknown value {advanced_strategy["type"]}')
