from models.network.deformable_detr_tracking.build_siamfc_multi_res_deform_atten_track import build_siamfc_multires_deform_atten_track
from Utils.yaml_config import load_config
import torch
from .tracker import DeformableTracker
from data.siamfc.processor.z_curate_x_resize import SiamFCZCurateXResizeProcessor


def build_deform_tracker(network_config: str, weight_path: str, device: str):
    device = torch.device(device)
    network_config = load_config(network_config)
    network = build_siamfc_multires_deform_atten_track(network_config)
    network.load_state_dict(torch.load(weight_path, map_location='cpu')['model'])
    processor = SiamFCZCurateXResizeProcessor(network_config['backbone']['siamfc']['exemplar_size'],
                                              network_config['backbone']['siamfc']['instance_size'],
                                              network_config['backbone']['siamfc']['context'])
    return DeformableTracker(network, device, processor)
