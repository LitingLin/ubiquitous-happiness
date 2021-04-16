import sys
import os
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(root_path)
default_config_path = os.path.join(root_path, 'config', 'transt')

import argparse
from pathlib import Path
import Utils.detr_misc as utils
from training.transt.training_loop import run_training_loop
from training.transt.builder import build_training_actor_and_dataloader

from Utils.yaml_config import load_config

from workarounds.all import apply_all_workarounds


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer tracker parameters', add_help=False)
    parser.add_argument('--network_config', type=str, default=os.path.join(default_config_path, 'config.yaml'), help='Path to the network config')
    parser.add_argument('--train_config', type=str, default=os.path.join(default_config_path, 'train.yaml'), help='Path to the train config')
    parser.add_argument('--train_dataset_config', type=str, default=os.path.join(default_config_path, 'dataset', 'train.yaml'), help='Path to the train dataset config')
    parser.add_argument('--val_dataset_config', type=str, default=os.path.join(default_config_path, 'dataset', 'val.yaml'), help='Path to the val dataset config')
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    print(args)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    apply_all_workarounds(seed)

    net_config = load_config(args.network_config)
    train_config = load_config(args.train_config)

    actor, train_data_loader, val_data_loader = build_training_actor_and_dataloader(args, net_config, train_config, args.train_dataset_config, args.val_dataset_config)
    run_training_loop(args, train_config, actor, train_data_loader, val_data_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('TransT training script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)
