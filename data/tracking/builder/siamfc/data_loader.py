


def _build_siamfc_tracking_data_loader(data_config: dict, dataset_config_path: str, post_processor, seed):



    pass




def build_siamfc_sampling_dataloader(train_config: dict, train_dataset_config_path: str, val_dataset_config_path: str, train_post_processor, val_post_processor):
    if 'version' not in train_config or train_config['version'] < 2:
        from data.siamfc.dataset import build_tracking_dataset
        return build_tracking_dataset(train_config, train_dataset_config_path, val_dataset_config_path, train_post_processor, val_post_processor)
    else:
        train_data_config = None
        if 'data' in train_config['train']:
            train_data_config = train_config['train']['data']

        train_dataset = _build_siamfc_tracking_data_loader(train_data_config, train_dataset_config_path, train_post_processor, 33)

        val_data_config = None
        if 'data' in train_config['val']:
            val_data_config = train_config['val']['data']

        eval_dataset = _build_siamfc_tracking_data_loader(val_data_config, val_dataset_config_path, val_post_processor, 44)

        return train_dataset, eval_dataset
