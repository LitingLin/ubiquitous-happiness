from data.torch.data_loader import build_torch_train_val_dataloader
from .dataset import build_siamfc_sampling_dataset


def build_siamfc_sampling_dataloader(args, train_config: dict, train_dataset_config_path: str,
                                     val_dataset_config_path: str, train_post_processor, val_post_processor,
                                     collate_fn, device_tensor_selection_filter):
    if 'version' not in train_config or train_config['version'] < 2:
        from data.siamfc.dataset import build_tracking_dataset
        return build_tracking_dataset(train_config, train_dataset_config_path, val_dataset_config_path,
                                      train_post_processor, val_post_processor)
    else:
        train_data_config = None
        if 'data' in train_config['train']:
            train_data_config = train_config['train']['data']

        train_dataset, train_worker_init_fn = build_siamfc_sampling_dataset(train_data_config, train_dataset_config_path, train_post_processor)

        val_data_config = None
        if 'data' in train_config['val']:
            val_data_config = train_config['val']['data']

        val_dataset, val_worker_init_fn = build_siamfc_sampling_dataset(val_data_config, val_dataset_config_path, val_post_processor)

        epoch_changed_event_signal_slots = []

        train_data_loader, val_data_loader = build_torch_train_val_dataloader(train_dataset, val_dataset,
                                                                              train_config['train']['batch_size'],
                                                                              train_config['val']['batch_size'],
                                                                              args.num_workers, args.num_workers,
                                                                              args.device, args.distributed,
                                                                              device_tensor_selection_filter,
                                                                              epoch_changed_event_signal_slots,
                                                                              train_worker_init_fn, val_worker_init_fn,
                                                                              collate_fn)

        return (train_dataset, val_dataset), (train_data_loader, val_data_loader), epoch_changed_event_signal_slots
