import torch
from models.TransT.network import build_transt
from algorithms.tracker.transt.tracker import TransTTracker


def build_transt_tracker(network_config, evaluation_config, weight_path, device):
    device = torch.device(device)
    model = build_transt(network_config, False)
    model.load_state_dict(torch.load(weight_path, map_location='cpu')['model'])

    return TransTTracker(model, device,
                         evaluation_config['tracking']['window_penalty'],
                         network_config['data']['template_size'], network_config['data']['search_size'],
                         network_config['data']['feature_size']['search'],
                         network_config['data']['area_factor']['template'],
                         network_config['data']['area_factor']['search'])
