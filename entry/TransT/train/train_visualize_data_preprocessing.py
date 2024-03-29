import sys
import os
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(root_path)
config_path = os.path.join(root_path, 'config', 'transt')

import argparse
from workarounds.all import apply_all_workarounds
from miscellanies.torch.print_running_environment import print_running_environment
from miscellanies.yaml_ops import yaml_load
from miscellanies.git_status import get_git_status_message
from training.transt._old.v3_builder import build_training_dataloader
from data.tracking.methods.TransT.training.label.viewer.builder import build_data_preprocessing_viewer
import copy


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer tracker parameters', add_help=False)
    parser.add_argument('config_name', type=str, help='Config name')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--persistent_data_workers', action='store_true', help='make the workers of dataloader persistent')
    parser.add_argument('--pin_memory', action='store_true', help='move tensors to pinned memory before transferring to GPU')
    parser.add_argument('--visualization_target', default='train', help='train or val')
    parser.add_argument('--batch_size', default=4, type=int, help='overwrite dataloader batch size')
    return parser


def main(args):
    seed = args.seed
    apply_all_workarounds(seed)

    print(f"git: {get_git_status_message()}")
    print_running_environment(args)

    print(args)

    network_config_path = os.path.join(config_path, args.config_name, 'config.yaml')
    train_config_path = os.path.join(config_path, args.config_name, 'train.yaml')
    train_dataset_config_path = os.path.join(config_path, args.config_name, 'dataset', 'train.yaml')
    val_dataset_config_path = os.path.join(config_path, args.config_name, 'dataset', 'val.yaml')
    network_config = yaml_load(network_config_path)
    train_config = yaml_load(train_config_path)

    args.distributed = False
    visualization_target = args.visualization_target
    assert visualization_target in ('train', 'val')
    train_config = copy.deepcopy(train_config)
    train_config['data']['sampler'][visualization_target]['batch_size'] = args.batch_size
    train_config['data']['with_raw_data'] = True

    _, (data_loader_train, data_loader_val), (stateful_objects, training_start_event_signal_slots, training_stop_event_signal_slots, epoch_changed_event_slots, statistics_collectors), stage_2_data_processor = \
        build_training_dataloader(args, network_config, train_config, train_dataset_config_path, val_dataset_config_path)
    if visualization_target == 'train':
        data_loader = data_loader_train
    else:
        data_loader = data_loader_val
    if training_start_event_signal_slots is not None:
        for signal in training_start_event_signal_slots:
            signal.start()
    viewer = build_data_preprocessing_viewer(data_loader, stage_2_data_processor, network_config, train_config, visualization_target)
    return viewer.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('TransT training script', parents=[get_args_parser()])
    args = parser.parse_args()
    sys.exit(main(args))
