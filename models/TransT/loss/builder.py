def build_criterion(train_config: dict):
    if 'version' not in train_config:
        from models.TransT.loss._old.builder import parse_old_transt_criterion_parameters
        loss_parameters = parse_old_transt_criterion_parameters(train_config)
    elif train_config['version'] == 2:
        loss_parameters = train_config['train']['loss']
        if 'use_template' in loss_parameters:
            from Miscellaneous.yaml_ops import yaml_load
            from Miscellaneous.repo_root import get_repository_root
            import os
            loss_parameters = yaml_load(os.path.join(get_repository_root(), 'config', 'transt', 'templates', 'loss', f"{loss_parameters['use_template']}.yaml"))
    elif train_config['version'] == 3:
        loss_parameters = train_config['optimization']['loss']
        if 'use_template' in loss_parameters:
            from Miscellaneous.yaml_ops import yaml_load
            from Miscellaneous.repo_root import get_repository_root
            import os
            loss_parameters = yaml_load(os.path.join(get_repository_root(), 'config', 'transt', 'templates', 'loss', f"{loss_parameters['use_template']}.yaml"))
    else:
        raise NotImplementedError(f"Unknown train config version {train_config['version']}")


    from .builders.cls_loss import build_cls_loss
    cls_loss, cls_loss_weight = build_cls_loss(loss_parameters)

    from .builders.bbox_loss import build_bbox_loss
    bbox_loss, bbox_loss_weight = build_bbox_loss(loss_parameters)

    from .builders.iou_loss import build_iou_loss
    iou_loss, iou_loss_weight = build_iou_loss(loss_parameters)

    loss_weight = {**cls_loss_weight, **bbox_loss_weight, **iou_loss_weight}

    from .transt import TransTCriterion
    return TransTCriterion(loss_weight, cls_loss, bbox_loss, iou_loss)
