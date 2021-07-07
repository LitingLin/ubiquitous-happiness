import time
from .train_step import train_step
from .eval_step import evaluate_step
from miscellanies.torch.checkpoint import dump_checkpoint_from_runner
import datetime


def print_model_efficiency_assessment(efficiency_assessor):
    print(efficiency_assessor.get_flop_count_table())

    init_time, track_time = efficiency_assessor.test_fps()
    batched_init_time, batched_track_time = efficiency_assessor.test_fps_batched()

    print(f"Estimated model FPS:\n@1: init {1/init_time} track {1/track_time}\n@{efficiency_assessor.get_batch()}: init {1/batched_init_time} track {1/batched_track_time}")


def run_training_loop(args, n_epochs, runner, logger, profiler, data_loader_train, data_loader_val, efficiency_assessor):
    with logger:
        print_model_efficiency_assessment(efficiency_assessor)
        logger.watch(runner.get_model())
        print("Start training")
        start_epoch = runner.get_epoch()
        start_time = time.perf_counter()
        with profiler, runner:
            for epoch in range(start_epoch, n_epochs):
                train_step(runner, logger, data_loader_train, args.logging_interval)
                evaluate_step(runner, logger, data_loader_val, args.logging_interval)

                runner.move_to_next_epoch()
                profiler.step()

                if args.output_dir is not None:
                    dump_checkpoint_from_runner(epoch, args.output_dir, runner, 10, args.checkpoint_interval)

            if args.output_dir is not None:
                dump_checkpoint_from_runner(epoch, args.output_dir, runner, 1, 1)
        total_time = time.perf_counter() - start_time
        total_time_str = str(datetime.timedelta(seconds=total_time))
        print('Training time {}'.format(total_time_str))
