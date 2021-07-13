from data.tracking.methods.TransT.evaluation.data_processor.transt import TransTEvaluationDataProcessor
from data.types.bounding_box_format import BoundingBoxFormat
from data.tracking.methods.TransT._common import _get_bounding_box_format, _get_bounding_box_normalization_helper


def build_siamfc_evaluation_data_processor(network_config, evaluation_config, device, preprocessing_on_device):
    from .data_processor.siamfc import SiamFCEvaluationDataProcessor
    from .post_processor.siamfc import SiamFCTrackingPostProcessing
    network_data_parameters = network_config['data']
    evaluation_parameters = evaluation_config['tracking']
    data_processor = SiamFCEvaluationDataProcessor(
        network_data_parameters['area_factor']['template'], network_data_parameters['area_factor']['search'],
        network_data_parameters['template_size'], network_data_parameters['search_size'],
        evaluation_parameters['scale_num'], evaluation_parameters['scale_step'], evaluation_parameters['scale_lr'],
        evaluation_parameters['min_size_factor'], evaluation_parameters['max_size_factor'],
        network_data_parameters['interpolation_mode'], network_data_parameters['imagenet_normalization'],
        device, preprocessing_on_device)
    network_post_processor = SiamFCTrackingPostProcessing(
        evaluation_parameters['response_up'], evaluation_parameters['scale_penalty'], evaluation_parameters['window_influence'],
        evaluation_parameters['response_norm'],
        network_config['head']['parameters']['size'], network_data_parameters['search_size'], device)
    return data_processor, network_post_processor


def build_evaluation_data_processors(network_config, evaluation_config, device):
    if network_config['version'] < 4:
        import data.tracking.methods.TransT.evaluation._old.builder
        return data.tracking.methods.TransT.evaluation._old.builder.build_evaluation_data_processors(network_config, evaluation_config, device)
    preprocessing_on_device = True
    if 'preprocessing_on_device' in evaluation_config['tracking']:
        preprocessing_on_device = evaluation_config['tracking']['preprocessing_on_device']
    print(f'Data preprocessing on Device: {preprocessing_on_device}')

    if network_config['type'].startswith('SiamFC'):
        return build_siamfc_evaluation_data_processor(network_config, evaluation_config, device, preprocessing_on_device)

    if 'bbox_size_limit_in_feat_space' not in evaluation_config['tracking']:
        bbox_size_limit_in_feat_space = False
    else:
        bbox_size_limit_in_feat_space = evaluation_config['tracking']['bbox_size_limit_in_feat_space']

    from data.tracking.methods.TransT.evaluation.bounding_box_post_processor.transt import \
        TransTBoundingBoxPostProcessor
    bounding_box_post_processor = TransTBoundingBoxPostProcessor(network_config['data']['search_size'], evaluation_config['tracking']['min_wh'], bbox_size_limit_in_feat_space, _get_bounding_box_normalization_helper(network_config), _get_bounding_box_format(network_config))

    data_processor = TransTEvaluationDataProcessor(
        network_config['data']['area_factor']['template'], network_config['data']['area_factor']['search'],
        network_config['data']['template_size'], network_config['data']['search_size'],
        network_config['data']['interpolation_mode'],
        device, preprocessing_on_device, bounding_box_post_processor)

    if network_config['head']['type'] == 'TransT':
        from data.tracking.methods.TransT.evaluation.post_processor.transt import TransTTrackingPostProcessing
        network_post_processor = TransTTrackingPostProcessing(network_config['head']['parameters']['input_size'], evaluation_config['tracking']['window_penalty'], device)
    elif network_config['head']['type'] == 'exp-1':
        from data.tracking.methods.TransT.evaluation.post_processor.exp_1 import TransTExp1TrackingPostProcessing
        network_post_processor = TransTExp1TrackingPostProcessing(evaluation_config['tracking']['window_penalty'] > 0, evaluation_config['tracking']['with_quality_assessment'], network_config['head']['parameters']['input_size'], device, evaluation_config['tracking']['window_penalty'])
    elif network_config['head']['type'] == 'GFocal-v2':
        from .post_processor.gfocal import GFocalTrackingPostProcessing
        network_post_processor = GFocalTrackingPostProcessing(evaluation_config['tracking']['window_penalty'] > 0, network_config['head']['parameters']['input_size'], device, evaluation_config['tracking']['window_penalty'])
    else:
        raise RuntimeError(f"Unknown value {network_config['head']['type']}")
    return data_processor, network_post_processor
