import time
from .train_step import train_one_epoch
from .eval_step import evaluate
from Miscellaneous.torch.distributed import is_main_process
from Miscellaneous.torch.checkpoint import _fail_safe_save, _fail_safe_copy
import os
import json
import datetime


def run_training_loop(args, train_config, actor, data_loader_train, data_loader_val):
    output_dir: str = args.output_dir

    print("Start training")
    start_time = time.perf_counter()
    for epoch in range(args.start_epoch, train_config['train']['epochs']):
        train_stats = train_one_epoch(actor, data_loader_train, epoch,
                                      train_config['train']['clip_max_norm'])
        actor.new_epoch()
        if output_dir and is_main_process():
            state_dict = actor.state_dict()
            latest_checkpoint_interval = 10
            backup_checkpoint_interval = args.checkpoint_interval

            saved_checkpoint_path = None
            if (epoch + 1) % latest_checkpoint_interval == 0:
                checkpoint_path = os.path.join(output_dir, 'checkpoint.pth')
                _fail_safe_save(checkpoint_path, state_dict)
                saved_checkpoint_path = checkpoint_path
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % backup_checkpoint_interval == 0:
                backup_checkpoint_path = os.path.join(output_dir, f'checkpoint{epoch:04}.pth')
                if saved_checkpoint_path is not None:
                    _fail_safe_copy(saved_checkpoint_path, backup_checkpoint_path)
                else:
                    _fail_safe_save(backup_checkpoint_path, state_dict)

        test_stats = evaluate(actor, data_loader_val)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': actor.n_parameters()}

        if args.output_dir and is_main_process():
            with open(os.path.join(output_dir, "log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    print('Training time {}'.format(total_time_str))
