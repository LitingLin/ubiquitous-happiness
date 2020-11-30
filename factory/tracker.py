import os
import torch


class TrackerFactory:
    tracker_list = {
        'siamfc-v1': {
            'name': 'siamfc-baseline-v1',
            'config_path': os.path.join(os.path.dirname(__file__), '..', 'config', 'siamfc', 'baseline-v1.yaml'),
            'weight_path': os.path.join(os.path.dirname(__file__), '..', 'weight', '2016-08-17.net.mat')
        },
        'siamfc-v1-gray': {
            'name': 'siamfc-baseline-v1',
            'config_path': os.path.join(os.path.dirname(__file__), '..', 'config', 'siamfc', 'baseline-v1.yaml'),
            'weight_path': os.path.join(os.path.dirname(__file__), '..', 'weight', '2016-08-17_gray025.net.mat')
        },
        'siamfc-v2': {
            'name': 'siamfc-baseline-v2',
            'config_path': os.path.join(os.path.dirname(__file__), '..', 'config', 'siamfc', 'baseline-v2.yaml'),
            'weight_path': os.path.join(os.path.dirname(__file__), '..', 'weight', 'baseline-conv5_e55.mat')
        },
        'siamfc-v2-gray': {
            'name': 'siamfc-baseline-v2',
            'config_path': os.path.join(os.path.dirname(__file__), '..', 'config', 'siamfc', 'baseline-v2.yaml'),
            'weight_path': os.path.join(os.path.dirname(__file__), '..', 'weight', 'baseline-conv5_gray_e100.mat')
        }
    }

    @staticmethod
    def list():
        return TrackerFactory.tracker_list.keys()

    @staticmethod
    def create(name: str = 'siamfc-v2', device: str = 'cuda:0'):
        from algorithms.tracker.siamfc.factory import build_siamfc_tracker
        return build_siamfc_tracker(device=torch.device(device), **TrackerFactory.tracker_list[name])
