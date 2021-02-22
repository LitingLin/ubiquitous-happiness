from models.network.detr_tracking_variants.encoder_shared_params_decoder_cross_attn_decoder_no_z_mask.builder import build_detr_tracking_network
from Utils.yaml_config import load_config
import torch
from .tracker import DETRTracker
from data.detr_tracking_variants.simple.evaluation_builder import build_evaluation_processor


def build_detr_tracker(network_config: str, weight_path: str, device: str):
    device = torch.device(device)
    network_config = load_config(network_config)
    network = build_detr_tracking_network(network_config)
    network.load_state_dict(torch.load(weight_path, map_location='cpu')['model'])
    processor = build_evaluation_processor(network_config)
    return DETRTracker(network_config['name'], network, device, processor)
