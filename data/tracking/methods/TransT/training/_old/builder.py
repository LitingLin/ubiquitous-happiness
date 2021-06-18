from data.tracking.methods.TransT.training.collate_fn import transt_collate_fn


def _build_transt_data_processor(network_config: dict, train_config: dict, label_generator):
    from data.tracking.methods.TransT.training.processor import TransTProcessor
    return TransTProcessor(network_config['data']['template_size'], network_config['data']['search_size'],
                           network_config['data']['area_factor']['template'],
                           network_config['data']['area_factor']['search'],
                           train_config['data']['scale_jitter_factor']['template'],
                           train_config['data']['scale_jitter_factor']['search'],
                           train_config['data']['translation_jitter_factor']['template'],
                           train_config['data']['translation_jitter_factor']['search'],
                           train_config['data']['gray_scale_probability'], train_config['data']['color_jitter'],
                           label_generator), transt_collate_fn


def build_transt_data_processor(network_config: dict, train_config: dict):
    if 'version' not in network_config or network_config['transformer']['head']['type'] == 'detr':
        from ..label.transt import TransTLabelGenerator
        label_generator = TransTLabelGenerator(network_config['data']['feature_size']['search'], network_config['data']['search_size'])
        return _build_transt_data_processor(network_config, train_config, label_generator)
    elif 'exp-1' in network_config['transformer']['head']['type']:
        from ..label.exp_1 import Exp1LabelGenerator
        label_generator = Exp1LabelGenerator(network_config['data']['feature_size']['search'], network_config['data']['search_size'], train_config['data']['gaussian_target_label_min_overlap'])
        return _build_transt_data_processor(network_config, train_config, label_generator)
    else:
        raise RuntimeError(f"Unknown {network_config['transformer']['head']['type']}")
