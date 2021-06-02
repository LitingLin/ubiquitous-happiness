def parse_old_transt_criterion_parameters(train_config: dict):
    from Miscellaneous.yaml_ops import yaml_load
    from Miscellaneous.repo_root import get_repository_root
    import os
    new_loss_parameters = yaml_load(os.path.join(get_repository_root(), 'config', 'transt', 'templates', 'loss', 'transt.yaml'))
    loss_parameters = train_config['train']['loss']
    new_loss_parameters['classification']['cross_entropy_loss']['background_class_weight'] = loss_parameters['eos_coef']
    new_loss_parameters['classification']['cross_entropy_loss']['weight'] = loss_parameters['weight']['cross_entropy']

    new_loss_parameters['bounding_box_regression']['IoU_loss']['weight'] = loss_parameters['weight']['giou']
    new_loss_parameters['bounding_box_regression']['L1_loss']['weight'] = loss_parameters['weight']['bbox']

    return new_loss_parameters