import time
from .train_step import train_one_epoch
from .eval_step import evaluate
from Miscellaneous.torch.checkpoint import dump_checkpoint_from_runner
import datetime
from fvcore.nn import FlopCountAnalysis, flop_count_table


def run_training_loop(args, n_epochs, runner, logger, profiler, data_loader_train, data_loader_val, pseudo_data_source):
    with logger:
        print(flop_count_table(FlopCountAnalysis(runner.get_model(), pseudo_data_source.get_train(1))))
        logger.watch(runner.get_model())
        print("Start training")
        start_epoch = runner.get_epoch()
        start_time = time.perf_counter()
        with profiler, runner:
            for epoch in range(start_epoch, n_epochs):
                train_one_epoch(runner, logger, data_loader_train, args.logging_interval)
                evaluate(runner, logger, data_loader_val, args.logging_interval)

                runner.move_to_next_epoch()
                profiler.step()

                if args.output_dir is not None:
                    dump_checkpoint_from_runner(epoch, args.output_dir, runner, 10, args.checkpoint_interval)

            if args.output_dir is not None:
                dump_checkpoint_from_runner(epoch, args.output_dir, runner, 1, 1)
        total_time = time.perf_counter() - start_time
        total_time_str = str(datetime.timedelta(seconds=total_time))
        print('Training time {}'.format(total_time_str))
