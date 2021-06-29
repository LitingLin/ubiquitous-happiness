import time
from .train_step import train_one_epoch
from .eval_step import evaluate
from Miscellaneous.torch.distributed import is_main_process
from Miscellaneous.torch.checkpoint import dump_checkpoint_from_runner
import os
import json
import datetime
from fvcore.nn import FlopCountAnalysis, flop_count_table


def _get_parameters_from_train_config(train_config: dict):
    if 'version' not in train_config or train_config['version'] < 3:
        epochs = train_config['train']['epochs']
        clip_max_norm = train_config['train']['clip_max_norm']
    else:
        epochs = train_config['optimization']['epochs']
        clip_max_norm = 0
        if 'clip_max_norm' in train_config['optimization']:
            clip_max_norm = train_config['optimization']['clip_max_norm']
    return epochs, clip_max_norm


def run_training_loop(args, n_epochs, runner, logger, profiler, data_loader_train, data_loader_val, pseudo_data_source):
    with logger:
        print(flop_count_table(FlopCountAnalysis(runner.get_model(), pseudo_data_source.get_train(1))))
        logger.watch(runner.get_model())
        print("Start training")
        start_epoch = runner.get_epoch()
        start_time = time.perf_counter()
        with profiler, runner:
            for epoch in range(start_epoch, n_epochs):
                train_stats = train_one_epoch(runner, logger, data_loader_train, args.logging_interval)
                test_stats = evaluate(runner, logger, data_loader_val, args.logging_interval)

                runner.move_to_next_epoch()
                profiler.step()

                if args.output_dir is not None:
                    dump_checkpoint_from_runner(epoch, args.output_dir, runner, 10, args.checkpoint_interval)
                    if is_main_process():
                        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                                     **{f'test_{k}': v for k, v in test_stats.items()},
                                     'epoch': epoch,
                                     'n_parameters': runner.n_parameters()}
                        with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                            f.write(json.dumps(log_stats) + "\n")

            if args.output_dir is not None:
                dump_checkpoint_from_runner(epoch, args.output_dir, runner, 1, 1)
        total_time = time.perf_counter() - start_time
        total_time_str = str(datetime.timedelta(seconds=total_time))
        print('Training time {}'.format(total_time_str))
