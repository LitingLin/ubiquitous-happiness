from data.torch.data_loader import build_torch_train_val_dataloader
from .dataset import build_siamfc_sampling_dataset
import numpy as np


def build_siamfc_sampling_dataloader(args, train_config: dict, train_dataset_config_path: str,
                                     val_dataset_config_path: str, train_post_processor, val_post_processor,
                                     collate_fn, device_tensor_selection_filter):
    seed = args.seed
    rng_engine = np.random.Generator(np.random.PCG64(seed))
    train_seed = rng_engine.integers(100, 1000000)
    val_seed = rng_engine.integers(100, 1000000)
    train_sampling_orchestrator_server_address = 'tcp://127.0.0.1:20005'
    val_sampling_orchestrator_server_address = 'tcp://127.0.0.1:30354'

    training_start_event_signal_slots = []
    training_stop_event_signal_slots = []
    stateful_objects = {}
    statistics_collectors = {}

    train_data_config = None
    if 'data' in train_config['train']:
        train_data_config = train_config['train']['data']

    train_sampler, train_dataset, train_worker_init_fn = build_siamfc_sampling_dataset(train_data_config, train_dataset_config_path,
                                                                        train_post_processor, train_sampling_orchestrator_server_address,
                                                                        train_seed, training_start_event_signal_slots, training_stop_event_signal_slots)

    stateful_objects['train_sampling_orchestrator'] = train_sampler
    statistics_collectors['train_sampling_orchestrator'] = train_sampler

    val_data_config = None
    if 'data' in train_config['val']:
        val_data_config = train_config['val']['data']

    val_sampler, val_dataset, val_worker_init_fn = build_siamfc_sampling_dataset(val_data_config, val_dataset_config_path,
                                                                    val_post_processor, val_sampling_orchestrator_server_address,
                                                                    val_seed, training_start_event_signal_slots, training_stop_event_signal_slots)

    stateful_objects['val_sampling_orchestrator'] = val_sampler
    statistics_collectors['val_sampling_orchestrator'] = val_sampler

    epoch_changed_event_signal_slots = []

    train_data_loader, val_data_loader = build_torch_train_val_dataloader(train_dataset, val_dataset,
                                                                          train_config['train']['batch_size'],
                                                                          train_config['val']['batch_size'],
                                                                          args.num_workers, args.num_workers,
                                                                          args.device, args.distributed,
                                                                          epoch_changed_event_signal_slots,
                                                                          False,
                                                                          device_tensor_selection_filter,
                                                                          train_worker_init_fn, val_worker_init_fn,
                                                                          collate_fn)

    return (train_dataset, val_dataset), (train_data_loader, val_data_loader), (stateful_objects, training_start_event_signal_slots, training_stop_event_signal_slots, epoch_changed_event_signal_slots, statistics_collectors)
