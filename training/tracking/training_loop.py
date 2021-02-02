import time
import torch
from .train_step import train_one_epoch
from .eval_step import evaluate
import Utils.detr_misc as utils
import os
import json
import datetime
import os
import shutil


def training_loop(args, train_config, actor, data_loader_train, data_loader_val):
    output_dir: str = args.output_dir
    if output_dir is not None and len(output_dir) > 0:
        os.makedirs(output_dir, exist_ok=True)

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, train_config['train']['epochs']):
        train_stats = train_one_epoch(actor, data_loader_train, epoch)
        actor.new_epoch()
        if output_dir:
            checkpoint_path = os.path.join(output_dir, 'checkpoint.pth')
            utils.save_on_master(actor.state_dict(), checkpoint_path)
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % train_config['train']['checkpoint_interval'] == 0:
                backup_checkpoint_path = os.path.join(output_dir, f'checkpoint{epoch:04}.pth')
                if os.path.exists(backup_checkpoint_path):
                    os.remove(backup_checkpoint_path)
                shutil.copy(checkpoint_path, backup_checkpoint_path)

        test_stats = evaluate(actor, data_loader_val)

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
