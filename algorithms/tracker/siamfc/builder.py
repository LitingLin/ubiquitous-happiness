from Utils.yaml_config import load_config
from .tracker import SiamFCTracker


def build_siamfc_tracker(name: str, config_path: str, weight_path: str, device):
    if name == 'siamfc-baseline-v1':
        from .v1.build import build_net
        network = build_net()
    elif name == 'siamfc-baseline-v2':
        from .v2.build import build_net
        network = build_net()
    else:
        raise Exception
    if weight_path.endswith('.mat'):
        from .weight_loader.matlab import load_matconvnet_weights
        load_matconvnet_weights(network, weight_path)
    else:
        from .weight_loader.torch import load_weights
        load_weights(network, weight_path)
    network.to(device)
    config = load_config(config_path)
    return SiamFCTracker(network.backbone, network.backbone, network.head, device, config['tracking'])
