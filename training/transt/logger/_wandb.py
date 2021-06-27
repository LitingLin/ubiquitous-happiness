try:
    import wandb
    has_wandb = True
except ImportError:
    has_wandb = False
from Miscellaneous.flatten_dict import flatten_dict
from Miscellaneous.git_status import get_git_status
from Miscellaneous.torch.distributed import is_main_process, is_dist_available_and_initialized


class WandbLogger:
    def __init__(self, id_, project_name: str, config: dict,
                 initial_step: int, log_freq: int,
                 only_log_on_main_process: bool,
                 watch_model_freq: int,
                 watch_model_parameters=False, watch_model_gradients=False,
                 ):
        self.id = id_
        self.project_name = project_name
        config = flatten_dict(config)
        config['git_version'] = get_git_status()
        self.tags = config['tags']
        self.config = config
        self.step = initial_step
        self.log_freq = log_freq
        self.only_log_on_main_process = only_log_on_main_process

        if watch_model_parameters and watch_model_gradients:
            watch_model = 'all'
        elif watch_model_parameters:
            watch_model = 'parameters'
        elif watch_model_gradients:
            watch_model = 'gradients'
        else:
            watch_model = None

        self.watch_model = watch_model
        self.watch_model_freq = watch_model_freq

    def __enter__(self):
        self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def start(self):
        if self.only_log_on_main_process and not is_main_process():
            return
        configs = {'project': self.project_name, 'entity': 'llt', 'tags': self.tags, 'config': flatten_dict(self.config),
                   'force': True, 'job_type': 'train', 'id': self.id}
        if not self.only_log_on_main_process and is_dist_available_and_initialized():
            configs['group'] = 'ddp'
        wandb.init(**configs)

    def log_train(self, epoch, forward_stats, backward_stats):
        if self.only_log_on_main_process and not is_main_process():
            return

        if self.step % self.log_freq == 0:
            log = {'epoch': epoch, 'batch': self.step, **forward_stats, **backward_stats}
            wandb.log(log, step=self.step)
        self.step += 1

    def log_test(self, epoch, summary):
        if self.only_log_on_main_process and not is_main_process():
            return

        if self.step % self.log_freq == 0:
            summary = {'test_' + k: v for k, v in summary.items()}
            summary['epoch'] = epoch

            wandb.log(summary, step=self.step)

    def watch(self, model):
        if self.only_log_on_main_process and not is_main_process():
            return
        wandb.watch(model, log=self.watch_model, log_freq=self.watch_model_freq)

    def stop(self):
        if self.only_log_on_main_process and not is_main_process():
            return
        wandb.finish()
