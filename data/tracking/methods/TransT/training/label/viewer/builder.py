import copy
from .main import DataPreprocessingVisualizer


def build_data_preprocessing_viewer(data_loader, stage_2_data_processor, network_config, train_config, visualization_target):
    head_type = network_config['head']['type']
    if head_type == 'TransT':
        from .transt import TransTDataPreprocessingVisualizer
        visualization_data_adaptor = TransTDataPreprocessingVisualizer(network_config, train_config, visualization_target)
    else:
        raise NotImplementedError(f'Unknown type {head_type}')

    return DataPreprocessingVisualizer(data_loader, stage_2_data_processor, visualization_data_adaptor)
