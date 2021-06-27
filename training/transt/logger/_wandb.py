try:
    import wandb
    has_wandb = True
except ImportError:
    has_wandb = False
from Miscellaneous.flatten_dict import flatten_dict
from Miscellaneous.git_status import get_git_status
from Miscellaneous.torch.distributed import is_main_process, is_dist_available_and_initialized


class WandbLogger:
    def __init__(self, id_, project_name: str, config: dict, only_log_on_main_process):
        self.id = id_
        self.project_name = project_name
        config = flatten_dict(config)
        config['git_version'] = get_git_status()
        self.tags = config['tags']
        self.config = config
        self.only_log_on_main_process = only_log_on_main_process

    def __enter__(self):
        self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def start(self):
        if self.only_log_on_main_process and not is_main_process():
            return
        configs = {'project': self.project_name, 'entity': 'llt', 'tags': self.tags, 'config': flatten_dict(self.config),
                   'force': True, 'job_type': 'train', 'id': self.id}
        if is_dist_available_and_initialized():
            configs['group'] = 'ddp'
        wandb.init(**configs)

    def log(self, epoch, batch, forward_stats, backward_stats):
        if self.only_log_on_main_process and not is_main_process():
            return

        log = {'batch': batch, **forward_stats, **backward_stats}

        wandb.log(log, step=epoch)

    def watch(self, model):
        if self.only_log_on_main_process and not is_main_process():
            return
        wandb.watch(model)

    def stop(self):
        if self.only_log_on_main_process and not is_main_process():
            return
        wandb.finish()
