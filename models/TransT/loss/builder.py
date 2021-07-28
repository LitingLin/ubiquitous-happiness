import importlib


def build_criterion(network_config: dict, train_config: dict, iterations_per_epoch: int):
    head_output_type = network_config['head']['output_protocol']
    loss_parameters = train_config['optimization']['loss']

    module_name_prefix = 'models.TransT.loss'

    head_loss_fn_module_prefix = module_name_prefix + '.' + head_output_type

    loss_modules = []

    if 'pre_filter' in loss_parameters:
        global_data_filter_build_function_module = importlib.import_module(head_loss_fn_module_prefix + '.pre_filter.' + loss_parameters['data_filter'])
        global_data_filter_build_function = getattr(global_data_filter_build_function_module, 'build_data_filter')
        global_data_filter = global_data_filter_build_function(train_config)
    else:
        global_data_filter = None

    for loss_cluster_module_name, loss_cluster_module_parameters in loss_parameters.items():
        loss_cluster_module_prefix = head_loss_fn_module_prefix + '.' + loss_cluster_module_name

        loss_functions = []
        loss_data_adaptor_functions = []
        loss_reduction_functions = []

        for loss_name, loss_parameters in loss_cluster_module_parameters.items():
            if loss_name in ('pre_filter', 'post_filter'):
                continue
            module = importlib.import_module(loss_cluster_module_prefix + '.' + loss_name)
            loss_function_build_function = getattr(module, 'build_' + loss_name)
            loss_function, loss_data_adaptor_function, loss_reduction_function = loss_function_build_function(loss_parameters, network_config, train_config)
            loss_functions.append(loss_function)
            loss_data_adaptor_functions.append(loss_data_adaptor_function)
            loss_reduction_functions.append(loss_reduction_function)

        if 'pre_filter' in loss_cluster_module_parameters:
            pre_sample_filter_parameters = loss_cluster_module_parameters['pre_filter']
            pre_sample_filter_function_type = pre_sample_filter_parameters['type']
            pre_sample_filter_function_module = loss_cluster_module_prefix + '.pre_filter.' + pre_sample_filter_function_type
            pre_sample_filter_function_build_function = getattr(importlib.import_module(pre_sample_filter_function_module), 'build_data_filter')
            pre_sample_filter_function = pre_sample_filter_function_build_function(pre_sample_filter_parameters)
        else:
            pre_sample_filter_function = None

        if 'post_filter' in loss_cluster_module_parameters:
            post_sample_filter_parameters = loss_cluster_module_parameters['post_filter']
            post_sample_filter_function_type = post_sample_filter_parameters['type']
            post_sample_filter_function_module = loss_cluster_module_prefix + '.post_filter.' + post_sample_filter_function_type
            post_sample_filter_function_build_function = getattr(importlib.import_module(post_sample_filter_function_module), 'build_data_filter')
            post_sample_filter_function = post_sample_filter_function_build_function(post_sample_filter_parameters)
        else:
            post_sample_filter_function = None

        loss_modules.append((loss_cluster_module_name, pre_sample_filter_function, loss_functions, loss_data_adaptor_functions, loss_reduction_functions, post_sample_filter_function))

    from .loss_compose.builder import build_loss_composer
    loss_composer = build_loss_composer(train_config, iterations_per_epoch)

    from .single_scale import SingleScaleCriterion
    criterion = SingleScaleCriterion(global_data_filter, loss_modules, loss_composer)

    return criterion, loss_composer
