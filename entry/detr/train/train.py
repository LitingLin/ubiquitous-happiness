from ...fix_path import *
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import datetime
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, DistributedSampler
import os
import Utils.detr_misc as utils
from training.detr.train_step import train_one_epoch
from training.detr.eval_step import evaluate

from Utils.yaml_config import load_config


from workarounds.all import apply_all_workarounds


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--net_config', type=str, default=None, help='Path to the net config')
    parser.add_argument('--train_config', type=str, default=None, help='Path to the train config')
    parser.add_argument('--train_dataset_config', type=str, help='Path to the train dataset config')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


def build_training_actor(args, net_config: dict, train_config: dict, num_classes: int, device):
    from models.network.detection.detr import build_detr_train
    from training.detr.actor import DETRActor
    model, criterion, postprocessors = build_detr_train(net_config, train_config, num_classes)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    train_backbone = train_config['train']['lr_backbone'] > 0
    for name, parameter in model.backbone.named_parameters():
        if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
            parameter.requires_grad_(False)

    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": train_config['train']['lr_backbone'],
        },
    ]

    optimizer = torch.optim.AdamW(param_dicts, lr=train_config['train']['lr'],
                                  weight_decay=train_config['train']['weight_decay'])
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, train_config['train']['lr_drop'])

    model.to(device)
    criterion.to(device)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    return DETRActor(model, criterion, optimizer, lr_scheduler), postprocessors


def build_dataloader(is_distributed: bool, num_workers: int, coco_path: str, batch_size: int):
    from Dataset.Detection.factory import DetectionDatasetFactory
    from Dataset.Detection.FactorySeeds.COCO import COCO_Seed, COCOVersion
    from Dataset.DataSplit import DataSplit
    from data.detr_wrapper.wrapper import DETRDataset
    from data.augmentation.detr import make_detr_transforms
    dataset_train = DetectionDatasetFactory(COCO_Seed(coco_path, data_split=DataSplit.Training, version=COCOVersion._2017)).construct()
    max_category_id = dataset_train.getMaxCategoryId()
    dataset_train = DETRDataset(dataset_train, make_detr_transforms('train'))

    dataset_val = DetectionDatasetFactory(COCO_Seed(coco_path, data_split=DataSplit.Validation, version=COCOVersion._2017)).construct()
    max_category_id = max(dataset_val.getNumberOfCategories(), max_category_id)
    dataset_val = DETRDataset(dataset_val, make_detr_transforms('val'))

    if is_distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=num_workers)
    data_loader_val = DataLoader(dataset_val, batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=num_workers)
    return data_loader_train, data_loader_val, max_category_id


def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    print(args)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    apply_all_workarounds(seed)

    device = torch.device(args.device)

    net_config = load_config(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'config', 'detr', 'network.yaml'), args.net_config)
    train_config = load_config(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'config', 'detr', 'train.yaml'), args.train_config)

    data_loader_train, data_loader_val, max_category_id = build_dataloader(args.distributed, args.num_workers, args.coco_path, train_config['train']['batch_size'])

    from pycocotools.coco import COCO
    coco_root = Path(args.coco_path)
    coco_val_annofile = coco_root / "annotations" / 'instances_val2017.json'
    coco_val_anno = COCO(coco_val_annofile)

    actor, postprocessors = build_training_actor(args, net_config, train_config, max_category_id, device)

    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        if args.eval:
            actor.load_state_dict(checkpoint, True)
        else:
            actor.load_state_dict(checkpoint, False)
            args.start_epoch = checkpoint['epoch'] + 1

    if args.eval:
        test_stats, coco_evaluator = evaluate(actor.model, actor.criterion, postprocessors,
                                              data_loader_val, coco_val_anno, device, args.output_dir)
        if args.output_dir:
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
        return

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, train_config['train']['epochs']):
        if args.distributed:
            data_loader_train.batch_sampler.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(actor, data_loader_train, device, epoch,
            train_config['train']['clip_max_norm'])
        actor.new_epoch()
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 100 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master(actor.state_dict(), checkpoint_path)

        test_stats, coco_evaluator = evaluate(
            actor.model, actor.criterion, postprocessors, data_loader_val, coco_val_anno, device, args.output_dir
        )

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': actor.n_parameters()}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

            # for evaluation logs
            if coco_evaluator is not None:
                (output_dir / 'eval').mkdir(exist_ok=True)
                if "bbox" in coco_evaluator.coco_eval:
                    filenames = ['latest.pth']
                    if epoch % 50 == 0:
                        filenames.append(f'{epoch:03}.pth')
                    for name in filenames:
                        torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                   output_dir / "eval" / name)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)
