import sys
import os
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(root_path)
config_path = os.path.join(root_path, 'config', 'transt')

from Miscellaneous.torch_print_running_environment import print_running_environment
from Utils.yaml_config import load_config
from algorithms.tracker.transt.builder import build_transt_tracker
from evaluation.SOT.runner import run_standard_evaluation, run_standard_report_generation
import Utils.detr_misc as utils


if __name__ == '__main__':
    from workarounds.all import apply_all_workarounds
    apply_all_workarounds()
    import argparse
    parser = argparse.ArgumentParser(description='Run tracker on OTB GOT10k LaSOT.')
    parser.add_argument('config_name', type=str, help='Config name')
    parser.add_argument('weight_path', type=str, help='Path to network weight')
    parser.add_argument('output_path', type=str, help="Path to save results.")
    parser.add_argument('--device', type=str, default='cuda:0', help="Pytorch device string.")
    parser.add_argument('--run-ope-evaluation-only', action='store_true', help="Run OPE evaluation only")
    parser.add_argument('--gen-report-only', action='store_true', help="Run report generation only")
    args = parser.parse_args()

    network_config_path = os.path.join(config_path, args.config_name, 'config.yaml')
    evaluation_config_path = os.path.join(config_path, args.config_name, 'evaluation.yaml')

    print("git:\n  {}\n".format(utils.get_sha()))
    print_running_environment(args)
    print(args)

    network_config = load_config(network_config_path)
    evaluation_config = load_config(evaluation_config_path)
    if args.gen_report_only:
        run_standard_report_generation(network_config['name'], args.output_path)
    else:
        tracker = build_transt_tracker(network_config, evaluation_config, args.weight_path, args.device)
        run_standard_evaluation(network_config['name'], tracker, args.output_path, not args.run_ope_evaluation_only)
