import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))

from factory.tracker import TrackerFactory
import argparse
from workarounds.all import apply_all_workarounds
from evaluation.experiment.run_vot import run_vot


def run_siamfc_on_vot(config_name: str = 'siamfc-v1-got', device: str = 'cuda:0'):
    apply_all_workarounds()
    tracker = TrackerFactory.create(config_name, device)
    run_vot(tracker)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run tracker on VOT.')
    parser.add_argument('--device', type=str, default='cuda:0', help="Pytorch device string.")
    args = parser.parse_args()
    run_siamfc_on_vot(args.device)
