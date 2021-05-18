import sys
import os

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(root_path)
config_path = os.path.join(root_path, 'config', 'transt')

import vot.tracker.trax
def vot_entry(arg_string):
    import os
    print(os.environ['PATH'])
    from algorithms.tracker.transt.build_from_args import build_from_arg_string
    tracker = build_from_arg_string(arg_string)
    from evaluation.SOT.protocol.vot.tracker_runner_rectangle import run_tracker
    run_tracker(tracker)


if __name__ == '__main__':
    from workarounds.all import apply_all_workarounds

    apply_all_workarounds()
    import argparse

    parser = argparse.ArgumentParser(description='Run tracker on several datasets.')
    parser.add_argument('vot_stack', type=str, help='VOT stack')
    parser.add_argument('config_name', type=str, help='Config name')
    parser.add_argument('weight_path', type=str, help='Path to network weight')
    parser.add_argument('output_path', type=str, help='VOT workspace')
    parser.add_argument('--evaluation-config-path', type=str, help='Path to evaluation config path.')
    parser.add_argument('--device', type=str, default='cuda:0', help="Pytorch device string.")
    args = parser.parse_args()

    network_config_path = os.path.join(config_path, args.config_name, 'config.yaml')

    from Miscellaneous.torch.print_running_environment import print_running_environment
    from Miscellaneous.yaml_ops import yaml_load
    from Miscellaneous.git_state import get_git_sha
    from evaluation.SOT.protocol.vot.prepare_workspace import prepare_vot_workspace
    from evaluation.SOT.protocol.vot.stack import VOTStack
    from evaluation.SOT.protocol.vot.vot_launcher import launch_vot_evaluation

    import subprocess

    print(f"git:\n  {get_git_sha()}\n")
    print_running_environment(args)
    print(args)

    network_config = yaml_load(network_config_path)

    vot_command = [args.config_name, args.weight_path, '--device', args.device]
    if args.evaluation_config_path is not None:
        vot_command += ['--evaluation-config-path', args.evaluation_config_path]
    parameter_string = subprocess.list2cmdline(vot_command).translate(str.maketrans({"'": r"\\'",
                                                                                     "\"": r'\\"',
                                                                                     "\\": r"\\\\"}))
    prepare_vot_workspace(args.output_path, network_config['name'],
                          f"from entry.TransT.test.test_vot import vot_entry; vot_entry('{parameter_string}')",
                          VOTStack[args.vot_stack])
    launch_vot_evaluation(args.output_path, network_config['name'])
