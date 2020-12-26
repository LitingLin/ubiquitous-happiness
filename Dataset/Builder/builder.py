import yaml
import os
import copy
import importlib
from Dataset.DataSplit import DataSplit


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


def forward_parameters(dataset, dataset_building_parameters):
    constructor = dataset.getConstructor()
    for parameter_name, parameter_value in dataset_building_parameters.items():
        if parameter_name in known_parameters:
            continue
        constructor.setDatasetAttribute(parameter_name, parameter_value)


def build_datasets(config: dict):
    filters = []
    if 'DATA_CLEANING' in config:
        dataset_filter_names = config['DATA_CLEANING']
        for filter_name in dataset_filter_names:
            module = importlib.import_module('Dataset.Filter.DataCleaner_{}'.format(filter_name))
            filter_class = getattr(module, 'DataCleaner_{}'.format(filter_name))
            filters.append(filter_class())
    if len(filters) == 0:
        filters = None
    datasets = []
    for dataset_name, dataset_building_parameter in config['DATASETS'].items():
        dataset_type = dataset_building_parameter['TYPE']
        path = None
        if 'PATH' in dataset_building_parameter:
            path = dataset_building_parameter['PATH']
        if dataset_type == 'DET':
            from Dataset.Detection.factory import DetectionDatasetFactory
            module = importlib.import_module('Dataset.Detection.FactorySeeds.{}'.format(dataset_name))
            seed_class = getattr(module, '{}_Seed'.format(dataset_name))
            seed = seed_class(root_path=path)
            seed.data_split = getDataSplitFromConfig(dataset_building_parameter['SPLITS'])
            factory = DetectionDatasetFactory(seed)
            dataset = factory.constructMemoryMappedVersion(filters)
            forward_parameters(dataset, dataset_building_parameter)
            datasets.append(dataset)
        elif dataset_type == 'SOT':
            from Dataset.SOT.factory import SingleObjectTrackingDatasetFactory
            module = importlib.import_module('Dataset.SOT.FactorySeeds.{}'.format(dataset_name))
            seed_class = getattr(module, '{}_Seed'.format(dataset_name))
            seed = seed_class(root_path=path)
            seed.data_split = getDataSplitFromConfig(dataset_building_parameter['SPLITS'])
            factory = SingleObjectTrackingDatasetFactory(seed)
            dataset = factory.constructMemoryMapped(filters)
            forward_parameters(dataset, dataset_building_parameter)
            datasets.append(dataset)
        elif dataset_type == 'MOT':
            from Dataset.MOT.factory import MultipleObjectTrackingDatasetFactory
            module = importlib.import_module('Dataset.MOT.FactorySeeds.{}'.format(dataset_name))
            seed_class = getattr(module, '{}_Seed'.format(dataset_name))
            seed = seed_class(root_path=path)
            seed.data_split = getDataSplitFromConfig(dataset_building_parameter['SPLITS'])
            factory = MultipleObjectTrackingDatasetFactory(seed)
            dataset = factory.construct(filters)
            forward_parameters(dataset, dataset_building_parameter)
            datasets.append(dataset)
        else:
            raise Exception('Unsupported dataset type {}'.format(dataset_type))
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
