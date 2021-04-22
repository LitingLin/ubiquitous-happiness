import torch
import torch.utils.data
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.dataloader import DataLoader
from data.performance.cuda_prefetcher import CUDAPrefetcher


def build_torch_train_val_dataloader(train_dataset, val_dataset,
                                     train_batch_size, val_batch_size,
                                     train_num_workers,
                                     val_num_workers,
                                     device, distributed, epoch_changed_event_signal_slots, collate_fn=None):
    do_shuffle = True
    if hasattr(train_dataset, 'set_epoch'):
        epoch_changed_event_signal_slots.append(train_dataset)
        do_shuffle = False
    if distributed:
        sampler_train = DistributedSampler(train_dataset, shuffle=do_shuffle)
        sampler_val = DistributedSampler(val_dataset, shuffle=False)
        epoch_changed_event_signal_slots.extend([sampler_train, sampler_val])
    else:
        sampler_train = torch.utils.data.RandomSampler(train_dataset)
        sampler_val = torch.utils.data.SequentialSampler(val_dataset)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, train_batch_size, drop_last=True)

    data_loader_train = DataLoader(train_dataset, batch_sampler=batch_sampler_train,
                                   num_workers=train_num_workers, collate_fn=collate_fn)
    data_loader_val = DataLoader(val_dataset, val_batch_size, sampler=sampler_val,
                                 drop_last=False, num_workers=val_num_workers, collate_fn=collate_fn)

    if 'cuda' in device:
        device = torch.device(device)
        data_loader_train = CUDAPrefetcher(data_loader_train, device)
        data_loader_val = CUDAPrefetcher(data_loader_val, device)

    return data_loader_train, data_loader_val
