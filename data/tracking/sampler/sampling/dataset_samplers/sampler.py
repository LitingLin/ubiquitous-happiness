import numpy as np


class TrackingDataset_SiamFCSampler:
    def __init__(self, datasets, negative_ratio, rng_engine=np.random):
        self.datasets = {}
        for dataset in datasets:
            pass

    def move_next(self):
        pass

    def get(self):
        pass


class SliceWrapper:
    def __init__(self, data_iterator, world_size, num_workers, drop_last=True):
        if world_size > 1:
            assert drop_last
        self.world_size = world_size
        self.num_workers = num_workers
        self.drop_last = drop_last
        self.rank_id = 0
        self.worker_id = 0
        self.data_iterator = data_iterator
        self.position = -1

    def set_rank_id(self, rank_id: int):
        self.rank_id = rank_id

    def set_worker_id(self, worker_id: int):
        self.worker_id = worker_id

    def __iter__(self):
        pass

    def __next__(self):
        while True:
            self.data_iterator.move_next()
            self.position += 1
            slice_length = self.num_workers * self.world_size
            slice_index = self.worker_id + self.rank_id * self.world_size
            if self.position % slice_length == slice_index:
                return self.data_iterator.get()

    def collate_fn(self, data: list):
        pass


class BatchWrapper:
    def __init__(self, batch_size):
        pass


class DatasetSampler:
    def __init__(self, dataset, dataset_sampler, sequence_sampler, track_sampler):
        self.dataset = dataset
        self.dataset_sampler = dataset_sampler
        self.sequence_sampler = sequence_sampler
        self.track_sampler = track_sampler

    def get_state(self):
        return self.dataset_sampler.get_state(), self.dataset_sampler.get_state(), self.dataset_sampler.get_state()

    def restore_from_state(self):
        pass

    def move_next(self):
        pass

    def current(self):
        sequence_index = self.dataset_sampler.current()
        track_index = self.sequence_sampler.current()
        object_ = self.track_sampler.current()
        return object_
