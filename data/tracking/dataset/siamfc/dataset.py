import torch.utils.data
import numpy as np
import torch


def siamfc_dataset_worker_init_fn(_, seed, dataset):
    dataset.seed_rng_engine(seed)


class SiamFCDataset(torch.utils.data.Dataset):
    def __init__(self, sampler, samples_per_epoch):
        self.dataset_sampler = sampler
        self.samples_per_epoch = samples_per_epoch
        self.rng_engine = np.random.Generator(np.random.PCG64())

    def seed_rng_engine(self, seed):
        self.rng_engine = np.random.Generator(np.random.PCG64(seed))

    def __getitem__(self, index):
        self.dataset_sampler.move_next(self.rng_engine)
        return self.dataset_sampler.do_sampling(self.rng_engine)

    def __len__(self):
        return self.samples_per_epoch
