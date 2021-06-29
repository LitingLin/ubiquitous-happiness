from data.torch.data_loader import build_torch_train_val_dataloader
from .dataset import build_siamfc_sampling_dataset


def build_siamfc_sampling_dataloader(args, train_config: dict, train_dataset_config_path: str,
                                     val_dataset_config_path: str, train_post_processor, val_post_processor,
                                     collate_fn, device_tensor_selection_filter):
    train_data_config = train_config['data']['sampler']['train']
    train_dataset, train_worker_init_fn = build_siamfc_sampling_dataset(train_data_config, train_dataset_config_path, train_post_processor)

    val_data_config = train_config['data']['sampler']['val']
    val_dataset, val_worker_init_fn = build_siamfc_sampling_dataset(val_data_config, val_dataset_config_path, val_post_processor)

    epoch_changed_event_signal_slots = []

    train_data_loader, val_data_loader = build_torch_train_val_dataloader(train_dataset, val_dataset,
                                                                          train_config['data']['sampler']['train']['batch_size'],
                                                                          train_config['data']['sampler']['val']['batch_size'],
                                                                          args.num_workers, args.num_workers,
                                                                          args.device, args.distributed,
                                                                          epoch_changed_event_signal_slots,
                                                                          False,
                                                                          device_tensor_selection_filter,
                                                                          train_worker_init_fn, val_worker_init_fn,
                                                                          collate_fn, args.pin_memory)

    return (train_dataset, val_dataset), (train_data_loader, val_data_loader), epoch_changed_event_signal_slots
