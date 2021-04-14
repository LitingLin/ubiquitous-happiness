from .processor import TransTProcessor


def build_transt_data_processor(network_config: dict, train_config: dict):
    return TransTProcessor(network_config['data']['template_size'], network_config['data']['search_size'],
                           network_config['data']['area_factor']['template'],
                           network_config['data']['area_factor']['search'],
                           train_config['data']['scale_jitter_factor']['template'],
                           train_config['data']['scale_jitter_factor']['search'],
                           train_config['data']['translation_jitter_factor']['template'],
                           train_config['data']['translation_jitter_factor']['search'],
                           train_config['data']['gray_scale_probability'], train_config['data']['brightness_factor'],
                           network_config['data']['feature_size']['search'])
