# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import os
from torch.utils.data import DataLoader
import Utils.detr_misc as utils
from detr.training.step import train_one_epoch
from detr.eval.step import evaluate

from Utils.yaml_config import load_config


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--net_config', type=str, default=None, help='Path to the net config')
    parser.add_argument('--train_config', type=str, default=None, help='Path to the train config')
    parser.add_argument('--train_dataset_config', type=str, help='Path to the train dataset config')
    parser.add_argument('--val_dataset_config', type=str, help='Path to the val dataset config')

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


from detr.training.actor.detr import build_detr_training_actor
from detr.data.sampler.siamfc import TrkDataset
from debugging.numpy_fix import apply_numpy_performance_fix
from debugging.cv2_fix import fix_strange_opencv_crash


def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    print(args)

    device = torch.device(args.device)

    net_config = load_config(os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'transformer', 'prototype1', 'network.yaml'), args.net_config)
    train_config = load_config(os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'transformer', 'prototype1', 'train.yaml'), args.train_config)
    train_dataset = TrkDataset(args.train_dataset_config, train_config['train']['samples_per_epoch'],
                               net_config['exemplar_size'], train_config['train']['image_size_limit'])
    val_dataset = TrkDataset(args.val_dataset_config, train_config['val']['samples_per_epoch'],
                             net_config['exemplar_size'], train_config['train']['image_size_limit'])

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    actor = build_detr_training_actor(args, net_config, train_config)
    actor.to(device)

    data_loader_train = DataLoader(train_dataset, batch_size=train_config['train']['batch_size'],
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers, worker_init_fn=worker_init_fn)
    data_loader_val = DataLoader(val_dataset, batch_size=train_config['val']['batch_size'],
                                 collate_fn=utils.collate_fn, num_workers=args.num_workers, worker_init_fn=worker_init_fn)

    output_dir = Path(args.output_dir)

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, train_config['train']['epochs']):
        train_stats = train_one_epoch(
            actor, data_loader_train, device, epoch,
            train_config['train']['clip_max_norm'])
        actor.new_epoch()
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % train_config['train']['lr_drop'] == 0 or (epoch + 1) % 100 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master(actor.state_dict(), checkpoint_path)

        test_stats = evaluate(
            actor, data_loader_val, device
        )

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': actor.n_parameters()}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    apply_numpy_performance_fix()
    fix_strange_opencv_crash()
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
