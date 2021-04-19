import sys
import os
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(root_path)
default_config_path = os.path.join(root_path, 'config', 'transt-swin')

if __name__ == '__main__':
    from workarounds.all import apply_all_workarounds

    apply_all_workarounds()
    import argparse

    parser = argparse.ArgumentParser(description='Run tracker on OTB GOT10k LaSOT.')
    parser.add_argument('weight_path', type=str, help='Path to network weight')
    parser.add_argument('output_path', type=str, help="Path to save results.")
    parser.add_argument('--network-config', type=str, default=os.path.join(default_config_path, 'config.yaml'),
                        help='Path to network config')
    parser.add_argument('--evaluation-config', type=str, default=os.path.join(default_config_path, 'evaluation.yaml'),
                        help='Path to evaluation config')
    parser.add_argument('--device', type=str, default='cuda:0', help="Pytorch device string.")
    args = parser.parse_args()

    import torch

    device = torch.device(args.device)

    from Miscellaneous.torch_print_running_environment import print_running_environment

    print_running_environment(device)

    from Utils.yaml_config import load_config
    from algorithms.tracker.transt.builder import build_transt_tracker
    from evaluation.SOT.runner import run_standard_evaluation

    network_config = load_config(args.network_config)
    evaluation_config = load_config(args.evaluation_config)
    tracker = build_transt_tracker(network_config, evaluation_config, args.weight_path, device)

    run_standard_evaluation(network_config['name'], tracker, args.output_path)
