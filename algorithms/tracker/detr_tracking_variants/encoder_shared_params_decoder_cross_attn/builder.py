from models.network.detr_tracking_variants.encoder_shared_params_decoder_cross_attn.builder import build_detr_tracking_network
from Utils.yaml_config import load_config
import torch
from .tracker import DETRTracker
from data.siamfc.processor.z_curate_x_size_limit import SiamFC_Z_Curate_BBOX_XYWH_X_SizeLimit_BBOX_CXCYWHNormalized_Processor


def build_detr_tracker(network_config: str, weight_path: str, device: str):
    device = torch.device(device)
    network_config = load_config(network_config)
    network = build_detr_tracking_network(network_config)
    network.load_state_dict(torch.load(weight_path, map_location='cpu')['model'])
    processor = SiamFC_Z_Curate_BBOX_XYWH_X_SizeLimit_BBOX_CXCYWHNormalized_Processor(network_config['backbone']['siamfc']['exemplar_size'],
                                                    network_config['backbone']['siamfc']['instance_size_limit'],
                                                    network_config['backbone']['siamfc']['context'])
    return DETRTracker(network, device, processor)
