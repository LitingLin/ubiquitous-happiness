import sys
import os
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(root_path)
config_path = os.path.join(root_path, 'config', 'transt')

import argparse
from pathlib import Path
from training.transt._old.training_loop import run_training_loop
from training.transt._old.builder import build_training_actor_and_dataloader
from workarounds.all import apply_all_workarounds
from Miscellaneous.torch.print_running_environment import print_running_environment
from Miscellaneous.yaml_ops import yaml_load
from Miscellaneous.git_state import get_git_sha
from Miscellaneous.torch.distributed import get_rank, init_distributed_mode


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer tracker parameters', add_help=False)
    parser.add_argument('config_name', type=str, help='Config name')
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--checkpoint_interval', default=100, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


def main(args):
    # fix the seed for reproducibility
    seed = args.seed + get_rank()
    apply_all_workarounds(seed)

    init_distributed_mode(args)
    print(f"git:\n  {get_git_sha()}\n")
    print_running_environment(args)

    print(args)

    network_config_path = os.path.join(config_path, args.config_name, 'config.yaml')
    train_config_path = os.path.join(config_path, args.config_name, 'train.yaml')
    train_dataset_config_path = os.path.join(config_path, args.config_name, 'dataset', 'train.yaml')
    val_dataset_config_path = os.path.join(config_path, args.config_name, 'dataset', 'val.yaml')
    network_config = yaml_load(network_config_path)
    train_config = yaml_load(train_config_path)

    actor, train_data_loader, val_data_loader = build_training_actor_and_dataloader(args, network_config, train_config, train_dataset_config_path, val_dataset_config_path)
    run_training_loop(args, train_config, actor, train_data_loader, val_data_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('TransT training script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)