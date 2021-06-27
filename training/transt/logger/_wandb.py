try:
    import wandb
    has_wandb = True
except ImportError:
    has_wandb = False
from Miscellaneous.flatten_dict import flatten_dict
from Miscellaneous.git_status import get_git_status


class LoggerWrapper:
    def before_run(self, project_name: str, tags: list, config: dict, ):
        config = flatten_dict(config)
        config['git_version'] = get_git_status()
        wandb.init(project=project_name, tags=tags, config=flatten_dict(config), force=True)