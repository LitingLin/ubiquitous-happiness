import time
from .train_step import train_one_epoch
from .eval_step import evaluate
from Miscellaneous.torch.distributed import is_main_process
from Miscellaneous.torch.checkpoint import dump_checkpoint_from_runner
import os
import json
import datetime


def run_training_loop(args, train_config, runner, data_loader_train, data_loader_val):
    print("Start training")
    start_time = time.perf_counter()
    for epoch in range(args.start_epoch, train_config['train']['epochs']):
        train_stats = train_one_epoch(runner, data_loader_train, epoch,
                                      train_config['train']['clip_max_norm'])
        test_stats = evaluate(runner, data_loader_val)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': runner.n_parameters()}

        runner.move_next_epoch()

        if args.output_dir is not None:
            dump_checkpoint_from_runner(epoch, args.output_dir, runner, 10, args.checkpoint_interval)
            if is_main_process():
                with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                    f.write(json.dumps(log_stats) + "\n")

    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    print('Training time {}'.format(total_time_str))
