import os
from Miscellaneous.repo_root import get_repository_root
from algorithms.tracker.transt.builder import build_transt_tracker
from Miscellaneous.yaml_ops import yaml_load
import shlex
_cache = {}


def build_from_arg_string(arg_string):
    print(arg_string)
    assert arg_string is not None
    if arg_string in _cache:
        return _cache[arg_string]

    import argparse
    parser = argparse.ArgumentParser(description='Build tracker')
    parser.add_argument('config_name', type=str, help='Config name')
    parser.add_argument('weight_path', type=str, help='Path to network weight')
    parser.add_argument('--evaluation-config-path', type=str, help='Path to evaluation config path.')
    parser.add_argument('--device', type=str, default='cuda:0', help="Pytorch device string.")
    args = parser.parse_args(shlex.split(arg_string))

    config_path = os.path.join(get_repository_root(), 'config', 'transt')

    network_config_path = os.path.join(config_path, args.config_name, 'config.yaml')
    evaluation_config_path = os.path.join(config_path, args.config_name, 'evaluation.yaml')

    if args.evaluation_config_path is not None:
        evaluation_config_path = args.evaluation_config_path

    network_config = yaml_load(network_config_path)
    evaluation_config = yaml_load(evaluation_config_path)
    tracker = build_transt_tracker(network_config, evaluation_config, args.weight_path, args.device)
    return tracker
