from Dataset.EasyBuilder.YAML.builder import build_datasets_from_yaml
from Miscellaneous.distributed import is_dist_available_and_initialized, is_main_process
import torch.distributed


def build_dataset_from_config_distributed_awareness(config_path: str):
    if not is_dist_available_and_initialized():
        return build_datasets_from_yaml(config_path)

    if is_main_process():
        datasets = build_datasets_from_yaml(config_path)

    torch.distributed.barrier()

    if not is_main_process():
        datasets = build_datasets_from_yaml(config_path)

    return datasets
