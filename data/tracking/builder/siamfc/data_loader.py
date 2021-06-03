import numpy as np
from data.randomness_control.iterable_dataset import IterableDatasetOrchestrator
from .dataset import build_siamfc_sampling_dataset
from Miscellaneous.torch.distributed import get_world_size, get_rank
import torch.utils.data.dataloader
from data.performance.cuda_prefetcher import CUDAPrefetcher
from data.randomness_control.ordered_batch_sampler import OrderedBatchSampler


def build_siamfc_sampling_dataloader(args, train_config: dict, train_dataset_config_path: str,
                                     val_dataset_config_path: str, train_post_processor, val_post_processor, seed: int,
                                     collate_fn):
    if 'version' not in train_config or train_config['version'] < 2:
        from data.siamfc.dataset import build_tracking_dataset
        return build_tracking_dataset(train_config, train_dataset_config_path, val_dataset_config_path,
                                      train_post_processor, val_post_processor)
    else:
        rank_id = get_rank()
        world_size = get_world_size()

        rng_engine = np.random.Generator(np.random.PCG64(seed))
        train_data_config = None
        if 'data' in train_config['train']:
            train_data_config = train_config['train']['data']

        train_dataset = build_siamfc_sampling_dataset(train_data_config, train_dataset_config_path,
                                                      train_post_processor, rng_engine.integers(0, 1000000))
        train_data_loader = IterableDatasetOrchestrator(train_dataset, train_config['train']['batch_size'], rank_id,
                                                        world_size,
                                                        np.random.SeedSequence(rng_engine.integers(0, 1000000)))
        torch_train_data_loader = torch.utils.data.dataloader.DataLoader(train_data_loader,
                                                                         batch_size=None,
                                                                         num_workers=args.num_workers)

        torch_train_data_loader = OrderedBatchSampler(torch_train_data_loader, train_config['train']['batch_size'], collate_fn)

        val_data_config = None
        if 'data' in train_config['val']:
            val_data_config = train_config['val']['data']

        val_dataset = build_siamfc_sampling_dataset(val_data_config, val_dataset_config_path, val_post_processor,
                                                    rng_engine.integers(0, 1000000))
        val_data_loader = IterableDatasetOrchestrator(val_dataset, train_config['val']['batch_size'], rank_id,
                                                      world_size,
                                                      np.random.SeedSequence(rng_engine.integers(0, 1000000)))
        torch_val_data_loader = torch.utils.data.dataloader.DataLoader(val_data_loader,
                                                                       batch_size=None,
                                                                       num_workers=args.num_workers)
        torch_val_data_loader = OrderedBatchSampler(torch_val_data_loader, train_config['val']['batch_size'], collate_fn)

        if 'cuda' in args.device:
            device = torch.device(args.device)
            torch_train_data_loader = CUDAPrefetcher(torch_train_data_loader, device)
            torch_val_data_loader = CUDAPrefetcher(torch_val_data_loader, device)

        return (train_data_loader, val_data_loader), (torch_train_data_loader, torch_val_data_loader)
