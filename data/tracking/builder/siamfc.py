
from data.distributed.dataset import build_dataset_from_config_distributed_awareness


def build_siamfc_tracking_data_loader(train_config: dict, train_dataset_config_path: str, val_dataset_config_path: str, train_post_processor, val_post_processor):
    if 'version' not in train_config or train_config['version'] < 2:
        from data.siamfc.dataset import build_tracking_dataset
        return build_tracking_dataset(train_config, train_dataset_config_path, val_dataset_config_path, train_post_processor, val_post_processor)
    else:
        raise NotImplementedError