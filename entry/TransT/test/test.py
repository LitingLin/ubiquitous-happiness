import sys
import os
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(root_path)
config_path = os.path.join(root_path, 'config', 'transt')

from Miscellaneous.torch.print_running_environment import print_running_environment
from Miscellaneous.yaml_ops import yaml_load
from Miscellaneous.git_state import get_git_sha
from algorithms.tracker.transt.builder import build_transt_tracker
from evaluation.SOT.runner import run_standard_evaluation, run_standard_report_generation


if __name__ == '__main__':
    from workarounds.all import apply_all_workarounds
    apply_all_workarounds()
    import argparse
    parser = argparse.ArgumentParser(description='Run tracker on OTB GOT10k LaSOT.')
    parser.add_argument('config_name', type=str, help='Config name')
    parser.add_argument('weight_path', type=str, help='Path to network weight')
    parser.add_argument('output_path', type=str, help="Path to save results.")
    parser.add_argument('--evaluation-config-path', type=str, help='Path to evaluation config path.')
    parser.add_argument('--device', type=str, default='cuda:0', help="Pytorch device string.")
    parser.add_argument('--run-ope-evaluation-only', action='store_true', help="Run OPE evaluation only")
    parser.add_argument('--gen-report-only', action='store_true', help="Run report generation only")
    args = parser.parse_args()

    network_config_path = os.path.join(config_path, args.config_name, 'config.yaml')
    evaluation_config_path = os.path.join(config_path, args.config_name, 'evaluation.yaml')

    if args.evaluation_config_path is not None:
        evaluation_config_path = args.evaluation_config_path

    print(f"git:\n  {get_git_sha()}\n")
    print_running_environment(args)
    print(args)

    network_config = yaml_load(network_config_path)
    evaluation_config = yaml_load(evaluation_config_path)
    if args.gen_report_only:
        run_standard_report_generation(network_config['name'], args.output_path)
    else:
        tracker = build_transt_tracker(network_config, evaluation_config, args.weight_path, args.device)
        run_standard_evaluation(network_config['name'], tracker, args.output_path, not args.run_ope_evaluation_only)
