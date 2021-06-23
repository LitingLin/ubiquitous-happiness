from data.tracking.methods.TransT.evaluation.data_preprocessor import TransTEvaluationDataProcessor
from data.tracking.methods.TransT.evaluation.bounding_box_post_processor import TransTBoundingBoxPostProcessor


def build_evaluation_data_processors(network_config, evaluation_config, device):
    if network_config['version'] < 4:
        import data.tracking.methods.TransT.evaluation._old.builder
        return data.tracking.methods.TransT.evaluation._old.builder.build_evaluation_data_processors(network_config, evaluation_config, device)
    preprocessing_on_device = True
    if 'preprocessing_on_device' in evaluation_config['tracking']:
        preprocessing_on_device = evaluation_config['tracking']['preprocessing_on_device']
    print(f'Data preprocessing on Device: {preprocessing_on_device}')

    if 'bbox_size_limit_in_feat_space' not in evaluation_config['tracking']:
        bbox_size_limit_in_feat_space = False
    else:
        bbox_size_limit_in_feat_space = evaluation_config['tracking']['bbox_size_limit_in_feat_space']

    bounding_box_post_processor = TransTBoundingBoxPostProcessor(network_config['data']['search_size'], evaluation_config['tracking']['min_wh'], bbox_size_limit_in_feat_space)

    data_processor = TransTEvaluationDataProcessor(
        network_config['data']['area_factor']['template'], network_config['data']['area_factor']['search'],
        network_config['data']['template_size'], network_config['data']['search_size'],
        network_config['data']['interpolation_mode'],
        device, preprocessing_on_device, bounding_box_post_processor)

    if network_config['head']['type'] == 'DETR':
        from data.tracking.methods.TransT.evaluation.post_processor.transt import TransTTrackingPostProcessing
        network_post_processor = TransTTrackingPostProcessing(network_config['head']['parameters']['input_size'], evaluation_config['tracking']['window_penalty'], device)
    elif network_config['head']['type'] == 'exp-1':
        from data.tracking.methods.TransT.evaluation.post_processor.exp_1 import TransTExp1TrackingPostProcessing
        network_post_processor = TransTExp1TrackingPostProcessing(evaluation_config['tracking']['window_penalty'] > 0, evaluation_config['tracking']['with_quality_assessment'], network_config['head']['parameters']['input_size'], device, evaluation_config['tracking']['window_penalty'])
    else:
        raise RuntimeError(f"Unknown value {network_config['head']['type']}")
    return data_processor, network_post_processor
