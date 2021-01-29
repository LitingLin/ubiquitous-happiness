import yaml
import os
import copy
import importlib
from Dataset.Type.data_split import DataSplit


def merge_config(default: dict, user_defined: dict):
    merged = {}
    for key, user_defined_value in user_defined.items():
        if user_defined_value is None:
            user_defined_value = {}
        if key in default:
            default_value = default[key]
            default_value = copy.copy(default_value)
            default_value.update(user_defined_value)
            user_defined_value = default_value
        merged[key] = user_defined_value

    return merged


def parseDataSplit(split_string: str):
    if split_string == 'train':
        return DataSplit.Training
    elif split_string == 'val':
        return DataSplit.Validation
    elif split_string == 'test':
        return DataSplit.Testing
    elif split_string == 'full':
        return DataSplit.Full
    else:
        raise Exception('Invalid value {}'.format(split_string))


def getDataSplitFromConfig(split_strings:list):
    split = parseDataSplit(split_strings[0])
    if len(split_strings) > 1:
        for split_string in split_strings[1:]:
            split |= parseDataSplit(split_string)
    return split


known_parameters = ['TYPE', 'SPLITS']


def forward_parameters(datasets, dataset_building_parameters):
    for dataset in datasets:
        manipulator = dataset.get_adhoc_manipulator()
        for parameter_name, parameter_value in dataset_building_parameters.items():
            if parameter_name in known_parameters:
                continue
            manipulator.set_attribute(parameter_name, parameter_value)


def build_datasets(config: dict):
    filters = []
    if 'FILTERS' in config:
        dataset_filter_names = config['FILTERS']
        for filter_key, filter_value in dataset_filter_names.items():
            if filter_key == 'DATA_CLEANING':
                for filter_data_cleaning_key, filter_data_cleaning_value in filter_value:
                    module = importlib.import_module('Dataset.Filter.DataCleaning.{}'.format(filter_data_cleaning_key))
                    filter_class = getattr(module, 'DataCleaning_{}'.format(filter_data_cleaning_key))
                    filters.append(filter_class(**filter_data_cleaning_value))
            else:
                module = importlib.import_module('Dataset.Filter.{}'.format(filter_key))
                filter_class = getattr(module, filter_key)
                filters.append(filter_class(**filter_value))

    if len(filters) == 0:
        filters = None
    datasets = []
    for dataset_name, dataset_building_parameter in config['DATASETS'].items():
        dataset_type = dataset_building_parameter['TYPE']
        path = None
        if 'PATH' in dataset_building_parameter:
            path = dataset_building_parameter['PATH']
        if dataset_type == 'SOT':
            from Dataset.SOT.factory import SingleObjectTrackingDatasetFactory
            module = importlib.import_module('Dataset.SOT.Seed.{}'.format(dataset_name))
            factory_class = SingleObjectTrackingDatasetFactory
        elif dataset_type == 'MOT':
            from Dataset.MOT.factory import MultipleObjectTrackingDatasetFactory
            module = importlib.import_module('Dataset.MOT.Seed.{}'.format(dataset_name))
            factory_class = MultipleObjectTrackingDatasetFactory
        elif dataset_type == 'DET':
            from Dataset.DET.factory import DetectionDatasetFactory
            module = importlib.import_module('Dataset.DET.Seed.{}'.format(dataset_name))
            factory_class = DetectionDatasetFactory
        else:
            raise Exception('Unsupported dataset type {}'.format(dataset_type))
        seed_class = getattr(module, '{}_Seed'.format(dataset_name))
        seed = seed_class(root_path=path)
        seed.data_split = getDataSplitFromConfig(dataset_building_parameter['SPLITS'])
        factory = factory_class([seed])
        dataset = factory.construct(filters)
        forward_parameters(dataset, dataset_building_parameter)
        datasets.extend(dataset)
    return datasets


def build_datasets_from_yaml(config_path: str):
    with open(os.path.join(os.path.dirname(__file__), 'dataset_def.yaml'), 'rb') as fid:
        default = yaml.safe_load(fid)
    default = default['DATASET_DEFINITIONS']
    with open(config_path, 'rb') as fid:
        config = yaml.safe_load(fid)
    dataset_config = merge_config(default, config['DATASETS'])
    config['DATASETS'] = dataset_config
    return build_datasets(config)
