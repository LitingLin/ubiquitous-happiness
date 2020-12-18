import time
import torch
from .train_step import train_one_epoch
from .eval_step import evaluate
import Utils.detr_misc as utils
import os
import json
import datetime


def training_loop(args, train_config, actor, data_loader_train, data_loader_val):
    output_dir: str = args.output_dir

    print("Start training")
    start_time = time.time()
    device = torch.device(args.device)
    for epoch in range(args.start_epoch, train_config['train']['epochs']):
        if args.distributed:
            data_loader_train.batch_sampler.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(actor, data_loader_train, device, epoch,
                                      train_config['train']['clip_max_norm'])
        actor.new_epoch()
        if output_dir:
            checkpoint_paths = [os.path.join(output_dir, 'checkpoint.pth')]
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % train_config['train']['lr_drop'] == 0 or (epoch + 1) % 100 == 0:
                checkpoint_paths.append(os.path.join(output_dir, f'checkpoint{epoch:04}.pth'))
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master(actor.state_dict(), checkpoint_path)

        test_stats = evaluate(actor, data_loader_val, device)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': actor.n_parameters()}

        if args.output_dir and utils.is_main_process():
            with open(os.path.join(output_dir, "log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
