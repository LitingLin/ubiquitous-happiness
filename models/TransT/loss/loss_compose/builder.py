def build_loss_composer(train_config: dict):
    from .composer import LinearWeightScheduler, ConstantWeightScheduler, LossComposer

    loss_parameters = train_config['optimization']['loss']

    weight_schedulers = []
    display_names = []
    for loss_parameter in loss_parameters.values():
        if isinstance(loss_parameter['weight'], dict):
            weight_parameters = loss_parameter['weight']
            assert weight_parameters['scheduler'] == 'linear'
            weight_scheduler = LinearWeightScheduler(weight_parameters['init'], weight_parameters['ultimate'],
                                                     0, weight_parameters['length'], weight_parameters['per_iter'])
        elif isinstance(loss_parameter['weight'], (int, float)):
            weight_scheduler = ConstantWeightScheduler(loss_parameter['weight'])
        else:
            raise NotImplementedError
        weight_schedulers.append(weight_scheduler)
        display_names.append(loss_parameter['disp_name'])
    return LossComposer(weight_schedulers, display_names)
