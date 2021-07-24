import torch
import torch.utils.data
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.dataloader import DataLoader


class _WorkerInitialization:
    def __init__(self, custom_init_fn):
        self.custom_init_fn = custom_init_fn

    def __call__(self, worker_id):
        from torch.utils.data import get_worker_info
        import numpy as np
        import random

        worker_info = get_worker_info()
        if worker_info is not None:
            worker_seed = worker_info.seed % 2 ** 32
            np.random.seed(worker_seed)
            random.seed(worker_seed)
            if self.custom_init_fn is not None:
                self.custom_init_fn(worker_id, worker_seed, worker_info.dataset)


def build_torch_train_val_dataloader(train_dataset, val_dataset,
                                     train_batch_size, val_batch_size,
                                     train_num_workers,
                                     val_num_workers,
                                     device, device_index, distributed,
                                     epoch_changed_event_signal_slots,
                                     training_do_shuffle=True,
                                     device_tensor_selection_filter=None,
                                     train_worker_init_fn=None, val_worker_init_fn=None,
                                     collate_fn=None, persistent_workers=False,
                                     pin_memory=False):
    if distributed:
        sampler_train = DistributedSampler(train_dataset, shuffle=training_do_shuffle)
        sampler_val = DistributedSampler(val_dataset, shuffle=False)
        epoch_changed_event_signal_slots.extend([sampler_train, sampler_val])
    else:
        if training_do_shuffle:
            sampler_train = torch.utils.data.RandomSampler(train_dataset)
        else:
            sampler_train = torch.utils.data.SequentialSampler(train_dataset)
        sampler_val = torch.utils.data.SequentialSampler(val_dataset)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, train_batch_size, drop_last=True)

    pin_memory = pin_memory and 'cuda' in device

    train_worker_initialization_object = _WorkerInitialization(train_worker_init_fn)
    val_worker_initialization_object = _WorkerInitialization(val_worker_init_fn)

    data_loader_train = DataLoader(train_dataset, batch_sampler=batch_sampler_train, worker_init_fn=train_worker_initialization_object,
                                   num_workers=train_num_workers, collate_fn=collate_fn, pin_memory=pin_memory, persistent_workers=persistent_workers)
    data_loader_val = DataLoader(val_dataset, val_batch_size, sampler=sampler_val, worker_init_fn=val_worker_initialization_object,
                                 drop_last=True, num_workers=val_num_workers, collate_fn=collate_fn, pin_memory=pin_memory, persistent_workers=persistent_workers)

    if 'cuda' in device:
        if pin_memory:
            from data.performance.cuda_prefetcher import CUDAPrefetcher
        else:
            from data.performance.cuda_prefetcher_thread import CUDAPrefetcher
        if device == 'cuda' and device_index is not None:
            device = f'cuda:{device_index}'
        device = torch.device(device)
        data_loader_train = CUDAPrefetcher(data_loader_train, device, device_tensor_selection_filter)
        data_loader_val = CUDAPrefetcher(data_loader_val, device, device_tensor_selection_filter)

    return data_loader_train, data_loader_val
