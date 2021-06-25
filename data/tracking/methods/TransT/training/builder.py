from data.tracking.methods.TransT.training.collate_fn import transt_collate_fn, SiamFC_collate_fn
from data.tracking.methods.TransT.training.processor_stage2 import TransTStage2DataProcessor


def _build_transt_data_processor(network_config: dict, train_config: dict, label_generator):
    from data.tracking.methods.TransT.training.processor import TransTProcessor

    network_data_config = network_config['data']
    train_data_augmentation_config = train_config['data']['augmentation']

    return TransTProcessor(network_data_config['template_size'], network_data_config['search_size'],
                           network_data_config['area_factor']['template'],
                           network_data_config['area_factor']['search'],
                           train_data_augmentation_config['scale_jitter_factor']['template'],
                           train_data_augmentation_config['scale_jitter_factor']['search'],
                           train_data_augmentation_config['translation_jitter_factor']['template'],
                           train_data_augmentation_config['translation_jitter_factor']['search'],
                           train_data_augmentation_config['gray_scale_probability'],
                           network_data_config['imagenet_normalization'],
                           train_data_augmentation_config['color_jitter'],
                           label_generator,
                           network_data_config['interpolation_mode'],
                           train_data_augmentation_config['stage_2_on_host_process'])


def build_transt_data_processor(network_config: dict, train_config: dict):
    if network_config['head']['type'] == 'DETR':
        from .label.transt import TransTLabelGenerator
        label_generator = TransTLabelGenerator(network_config['head']['parameters']['input_size'], network_config['data']['search_size'])
        return _build_transt_data_processor(network_config, train_config, label_generator), transt_collate_fn
    elif 'exp-1' in network_config['head']['type']:
        from .label.exp_1 import Exp1LabelGenerator
        label_generator = Exp1LabelGenerator(network_config['head']['parameters']['input_size'], network_config['data']['search_size'], train_config['data']['gaussian_target_label_min_overlap'])
        return _build_transt_data_processor(network_config, train_config, label_generator), transt_collate_fn
    elif network_config['head']['type'] == 'Stark':
        pass
    elif network_config['head']['type'] == 'SiamFC':
        from .label.siamfc import SiamFCLabelGenerator
        head_parameters = network_config['head']['parameters']
        label_generator = SiamFCLabelGenerator(head_parameters['size'], head_parameters['r_pos'], head_parameters['r_neg'], head_parameters['total_stride'])
        return _build_transt_data_processor(network_config, train_config, label_generator), SiamFC_collate_fn
    else:
        raise RuntimeError(f"Unknown {network_config['transformer']['head']['type']}")


def build_stage_2_data_processor(network_config: dict, train_config: dict):
    network_data_config = network_config['data']
    train_data_augmentation_config = train_config['data']['augmentation']

    if train_data_augmentation_config['stage_2_on_host_process']:
        return TransTStage2DataProcessor(network_data_config['template_size'], network_data_config['search_size'], train_data_augmentation_config['color_jitter'], network_data_config['interpolation_mode'], network_data_config['imagenet_normalization'])
    else:
        return None
