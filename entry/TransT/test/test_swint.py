import sys
import os
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(root_path)
default_config_path = os.path.join(root_path, 'config', 'transt-swin')

from algorithms.tracker.transt.builder import build_transt_tracker
from evaluation.SOT.runner import run_standard_evaluation
from workarounds.all import apply_all_workarounds
import argparse
from Utils.yaml_config import load_config


if __name__ == '__main__':
    apply_all_workarounds()
    parser = argparse.ArgumentParser(description='Run tracker on OTB GOT10k LaSOT.')
    parser.add_argument('weight_path', type=str, help='Path to network weight')
    parser.add_argument('output_path', type=str, help="Path to save results.")
    parser.add_argument('--network-config', type=str, default=os.path.join(default_config_path, 'config.yaml'),
                        help='Path to network config')
    parser.add_argument('--evaluation-config', type=str, default=os.path.join(default_config_path, 'evaluation.yaml'),
                        help='Path to evaluation config')
    parser.add_argument('--device', type=str, default='cuda:0', help="Pytorch device string.")
    args = parser.parse_args()

    network_config = load_config(args.network_config)
    evaluation_config = load_config(args.evaluation_config)

    tracker = build_transt_tracker(network_config, evaluation_config, args.weight_path, args.device)

    result_path = os.path.join(args.output_path, 'results')
    os.makedirs(result_path, exist_ok=True)
    report_path = os.path.join(args.output_path, 'reports')
    os.makedirs(report_path, exist_ok=True)

    run_standard_evaluation(network_config['name'], tracker, args.output_path)
