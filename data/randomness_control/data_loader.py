from torch.utils.data.dataset import Dataset
import torch.distributed as dist
import random
import numpy as np
import torch
import pickle

def _seed_all_rng_engine(seed: int):
    random.seed(seed)
    np.random.seed(random.randbytes(32))
    torch.manual_seed(random.randbytes(32))

def _is_dist_available_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def _get_rng_state():
    return {'py': random.getstate(),
            'np': np.random.get_state(),
            'torch': torch.get_rng_state()}

def _get_binary_rng_state():
    state = _get_rng_state()
    return pickle.dumps(state)

def _load_rng_state(state: dict):
    random.setstate(state['py'])
    np.random.set_state(state['np'])
    torch.set_rng_state(state['torch'])

def _load_binary_rng_state(state: bytes):
    state = pickle.loads(state)
    _load_rng_state(state)

class DataLoaderRandomnessOrchestrator(Dataset):
    def __init__(self, dataset: Dataset, batch_size: int, shuffle=False, seed: int=0):
        self.epoch = 0
        self.dataset = dataset
        self.seed = seed

        self.indices = None
        if shuffle:
            self._generate_random_indices()

        rng_state_bucket_size = 96*1024 # 96KB
        self.rng_state_buckets = torch.zeros([batch_size, rng_state_bucket_size], dtype=torch.uint8)
        self.rng_state_buckets.share_memory_()
        origin_state = _get_rng_state()
        for i in range(batch_size):
            _seed_all_rng_engine(seed + 100 * i)
            self._save_rng_state_to_bucket(i)

        _load_rng_state(origin_state)

    def _generate_random_indices(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        self.indices = torch.randperm(len(self.dataset), generator=g)
        self.indices.share_memory_()

    def set_epoch(self, epoch: int):
        self.epoch = epoch
        if self.indices is not None:
            self._generate_random_indices()

    def __len__(self):
        return len(self.dataset)

    def _load_rng_state_from_bucket(self, index: int):
        rng_state_bucket_size = self.rng_state_buckets.shape[1]

        bucket = memoryview(self.rng_state_buckets[index:].numpy())
        rng_state_size = int.from_bytes(bucket[0:4], byteorder='little', signed=False)
        assert rng_state_size <= rng_state_bucket_size - 4
        _load_binary_rng_state(bucket[4: 4 + rng_state_size])

    def _save_rng_state_to_bucket(self, index: int):
        rng_state_bucket_size = self.rng_state_buckets.shape[1]

        bucket = memoryview(self.rng_state_buckets[index:].numpy())
        rng_state = _get_binary_rng_state()
        rng_state_size = len(rng_state)
        bucket[0:4] = rng_state_size.to_bytes(4, byteorder='little', signed=False)

        assert rng_state_size <= rng_state_bucket_size - 4
        bucket[4: rng_state_size + 4] = rng_state

    def __getitem__(self, index: int):
        origin_rng_state = _get_rng_state()
        batch_size = self.rng_state_buckets.shape[0]

        rng_state_bucket_index = index % batch_size
        self._load_rng_state_from_bucket(rng_state_bucket_index)

        if self.indices is not None:
            index = self.indices[index]
        data = self.dataset[index]
        self._save_rng_state_to_bucket(rng_state_bucket_index)
        _load_rng_state(origin_rng_state)
        return data
