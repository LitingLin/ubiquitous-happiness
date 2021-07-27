import importlib


def loss_mean_reduction_function(loss):
    return loss.mean()


def loss_sum_reduction_function(loss):
    return loss.sum()


def build_criterion(network_config: dict, train_config: dict, iterations_per_epoch: int):
    head_output_type = network_config['head']['output_type']
    loss_parameters = train_config['optimization']['loss']

    module_name_prefix = 'models.TransT.loss'

    head_loss_fn_module_prefix = module_name_prefix + '.' + head_output_type

    loss_modules = []

    for loss_cluster_module_name, loss_cluster_module_parameters in loss_parameters.items():
        loss_cluster_module_prefix = head_loss_fn_module_prefix + '.' + loss_cluster_module_name

        loss_functions = []
        loss_data_adaptor_functions = []

        for loss_name, loss_parameters in loss_cluster_module_parameters.items():
            if loss_name == 'reduction' or loss_name == 'sample_filter':
                continue
            module = importlib.import_module(loss_cluster_module_prefix + '.' + loss_name)
            loss_function_build_function = getattr(module, 'build_' + loss_name)
            loss_function, loss_data_adaptor_function = loss_function_build_function(train_config)
            loss_functions.append(loss_function)
            loss_data_adaptor_functions.append(loss_data_adaptor_function)

        if 'sample_filter' in loss_cluster_module_parameters:
            sample_filter_parameters = loss_cluster_module_parameters['sample_filter']
            sample_filter_function_type = sample_filter_parameters['type']
            sample_filter_function_module = loss_cluster_module_prefix + '.sample_filter.' + sample_filter_function_type
            sample_filter_function_build_function = getattr(importlib.import_module(sample_filter_function_module), 'build_sample_filter')
            sample_filter_function = sample_filter_function_build_function(sample_filter_parameters)
        else:
            sample_filter_function = None

        if 'reduction' not in loss_cluster_module_parameters:
            loss_reduction_function = loss_mean_reduction_function
        else:
            loss_reduction_function_parameters = loss_cluster_module_parameters['reduction']
            loss_reduction_function_type = loss_reduction_function_parameters['type']
            if loss_reduction_function_type == 'mean':
                loss_reduction_function = loss_mean_reduction_function
            elif loss_reduction_function_type == 'sum':
                loss_reduction_function = loss_sum_reduction_function
            else:
                loss_reduction_function_module = loss_cluster_module_prefix + '.reduction.' + loss_reduction_function_type
                loss_reduction_function_builder = getattr(importlib.import_module(loss_reduction_function_module), 'build_loss_reduction_function')
                loss_reduction_function = loss_reduction_function_builder(loss_reduction_function_parameters)

        loss_modules.append((loss_cluster_module_name, sample_filter_function, loss_functions, loss_data_adaptor_functions, loss_reduction_function))

    from .single_scale import SingleScaleCriterion
    criterion = SingleScaleCriterion(loss_modules)

    from .loss_compose.builder import build_loss_composer
    loss_composer = build_loss_composer(train_config, iterations_per_epoch)

    return criterion, loss_composer
