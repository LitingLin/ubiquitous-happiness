import torch
from models.TransT.builder import build_transt
from algorithms.tracker.transt.tracker import TransTTracker


def build_transt_tracker(network_config, evaluation_config, weight_path, device):
    device = torch.device(device)
    model = build_transt(network_config, False)
    state_dict = torch.load(weight_path, map_location='cpu')['model']
    for key in list(state_dict.keys()):
        key: str = key
        if key.startswith('head.class_embed'):
            state_dict[key.replace('head.class_embed', 'head.classification')] = state_dict.pop(key)
        elif key.startswith('head.bbox_embed'):
            state_dict[key.replace('head.bbox_embed', 'head.regression')] = state_dict.pop(key)
    model.load_state_dict(state_dict)
    if 'bbox_size_limit_in_feat_space' not in evaluation_config['tracking']:
        bbox_size_limit_in_feat_space = False
    else:
        bbox_size_limit_in_feat_space = evaluation_config['tracking']['bbox_size_limit_in_feat_space']

    if 'version' not in network_config or network_config['transformer']['head']['type'] == 'detr':
        from .post_processor.transt import TransTTrackingPostProcessing
        post_processor = TransTTrackingPostProcessing(network_config['data']['feature_size']['search'], evaluation_config['tracking']['window_penalty'], device)
    elif network_config['transformer']['head']['type'] == 'exp-1':
        from .post_processor.exp_1 import TransTExp1TrackingPostProcessing
        post_processor = TransTExp1TrackingPostProcessing(evaluation_config['tracking']['window_penalty'] > 0, evaluation_config['tracking']['with_quality_assessment'], network_config['data']['feature_size']['search'], device, evaluation_config['tracking']['window_penalty'])
    else:
        raise RuntimeError(f"Unknown value {network_config['transformer']['head']['type']}")

    return TransTTracker(model, device, post_processor,
                         evaluation_config['tracking']['min_wh'],
                         network_config['data']['template_size'], network_config['data']['search_size'],
                         network_config['data']['feature_size']['search'],
                         network_config['data']['area_factor']['template'],
                         network_config['data']['area_factor']['search'],
                         bbox_size_limit_in_feat_space)
