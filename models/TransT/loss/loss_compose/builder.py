def build_loss_composer(train_config: dict, iterations_per_epoch: int):
    from .composer import LinearWeightScheduler, ConstantWeightScheduler, LossComposer

    loss_parameters = train_config['optimization']['loss']
    epochs = train_config['optimization']['epochs']
    total_iterations = epochs * iterations_per_epoch

    weight_schedulers = []
    display_names = []
    for cluster_loss_parameter in loss_parameters.values():
        for loss_name, loss_parameter in cluster_loss_parameter.items():
            if loss_name in ('pre_filter', 'post_filter'):
                continue
            if isinstance(loss_parameter['weight'], dict):
                weight_parameters = loss_parameter['weight']
                assert weight_parameters['scheduler'] == 'linear'
                weight_scheduler = LinearWeightScheduler(weight_parameters['initial_value'],
                                                         weight_parameters['ultimate_value'],
                                                         0, int(round(weight_parameters['length'] * total_iterations)),
                                                         weight_parameters['per_iteration'])
            elif isinstance(loss_parameter['weight'], (int, float)):
                weight_scheduler = ConstantWeightScheduler(loss_parameter['weight'])
            else:
                raise NotImplementedError
            weight_schedulers.append(weight_scheduler)
            display_names.append(loss_parameter['display_name'])
    return LossComposer(weight_schedulers, display_names)
