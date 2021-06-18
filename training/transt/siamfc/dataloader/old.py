from data.tracking.methods.TransT.training._old.builder import build_transt_data_processor
from data.performance.cuda_prefetcher import TensorFilteringByIndices


def build_old_dataloader(args, network_config: dict, train_config: dict, train_dataset_config_path: str,
                      val_dataset_config_path: str):
    processor, collate_fn = build_transt_data_processor(network_config, train_config)

    from data.siamfc.dataset import build_tracking_dataset
    from data.torch.data_loader import build_torch_train_val_dataloader
    train_dataset, val_dataset = build_tracking_dataset(train_config, train_dataset_config_path, val_dataset_config_path, processor, processor)

    epoch_changed_event_signal_slots = []

    data_loader_train, data_loader_val = build_torch_train_val_dataloader(train_dataset, val_dataset,
                                                                          train_config['data']['sampler']['train']['batch_size'],
                                                                          train_config['data']['sampler']['val']['batch_size'],
                                                                          args.num_workers, args.num_workers,
                                                                          args.device, args.distributed,
                                                                          epoch_changed_event_signal_slots,
                                                                          True,
                                                                          TensorFilteringByIndices((0, 1)),
                                                                          collate_fn=collate_fn)

    return (train_dataset, val_dataset), (data_loader_train, data_loader_val), epoch_changed_event_signal_slots
