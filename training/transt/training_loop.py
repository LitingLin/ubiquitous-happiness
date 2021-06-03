import time
from .train_step import train_one_epoch
from .eval_step import evaluate
from Miscellaneous.torch.distributed import is_main_process
from Miscellaneous.torch.checkpoint import dump_checkpoint
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
        data_loader_train.dataset.synchronize(epoch)
        test_stats = evaluate(actor, data_loader_val)
        data_loader_val.dataset.synchronize(epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': actor.n_parameters()}

        if output_dir and is_main_process():
            model_state_dict, training_state_dict = actor.state_dict()
            training_state_dict['train_data_loader'] = data_loader_train.dataset.get_state()
            training_state_dict['val_data_loader'] = data_loader_val.dataset.get_state()
            dump_checkpoint(epoch, args.output_dir, model_state_dict, training_state_dict, 10, args.checkpoint_interval)

        if args.output_dir and is_main_process():
            with open(os.path.join(output_dir, "log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    print('Training time {}'.format(total_time_str))
