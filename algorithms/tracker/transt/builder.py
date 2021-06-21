import torch
from models.TransT.builder import build_transt
from algorithms.tracker.transt.tracker import TransTTracker
from data.tracking.methods.TransT.evaluation._old.builder import build_evaluation_data_processors


def build_transt_tracker(network_config, evaluation_config, weight_path, device):
    device = torch.device(device)
    model = build_transt(network_config, False)
    state_dict = torch.load(weight_path, map_location='cpu')['model']

    if network_config['version'] <= 2:
        for key in list(state_dict.keys()):
            key: str = key
            if key.startswith('head.class_embed'):
                state_dict[key.replace('head.class_embed', 'head.classification')] = state_dict.pop(key)
            elif key.startswith('head.bbox_embed'):
                state_dict[key.replace('head.bbox_embed', 'head.regression')] = state_dict.pop(key)
        if network_config['backbone']['type'] == 'swin_transformer':
            from models.backbone.swint.swin_transformer import _update_state_dict_
            _update_state_dict_(state_dict, 'backbone.backbone.')

    model.load_state_dict(state_dict)

    data_processor, network_post_processor = build_evaluation_data_processors(network_config, evaluation_config, device)

    return TransTTracker(model, device, data_processor, network_post_processor)
