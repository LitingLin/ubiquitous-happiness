from .collate_fn import transt_collate_fn


def build_transt_data_processor(network_config: dict, train_config: dict):
    if 'version' not in network_config or network_config['transformer']['head']['type'] == 'detr':
        from .processor.transt import TransTProcessor
        return TransTProcessor(network_config['data']['template_size'], network_config['data']['search_size'],
                               network_config['data']['area_factor']['template'],
                               network_config['data']['area_factor']['search'],
                               train_config['data']['scale_jitter_factor']['template'],
                               train_config['data']['scale_jitter_factor']['search'],
                               train_config['data']['translation_jitter_factor']['template'],
                               train_config['data']['translation_jitter_factor']['search'],
                               train_config['data']['gray_scale_probability'], train_config['data']['color_jitter'],
                               network_config['data']['feature_size']['search']), transt_collate_fn
    elif 'exp-1' in network_config['transformer']['head']['type']:
        from .processor.exp_1 import TransTExp1Processor
        return TransTExp1Processor(network_config['data']['template_size'], network_config['data']['search_size'],
                                   network_config['data']['area_factor']['template'],
                                   network_config['data']['area_factor']['search'],
                                   train_config['data']['scale_jitter_factor']['template'],
                                   train_config['data']['scale_jitter_factor']['search'],
                                   train_config['data']['translation_jitter_factor']['template'],
                                   train_config['data']['translation_jitter_factor']['search'],
                                   train_config['data']['gray_scale_probability'], train_config['data']['color_jitter'],
                                   network_config['data']['feature_size']['search'],
                                   train_config['data']['gaussian_target_label_min_overlap']), transt_collate_fn
    else:
        raise RuntimeError(f"Unknown {network_config['transformer']['head']['type']}")
