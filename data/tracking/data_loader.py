import torch.utils.data.dataset
from enum import Enum, auto
import copy


class SliceManager:
    class Action(Enum):
        Do = auto()
        Skip = auto()
        Stop = auto()

    def __init__(self, default_length):
        self.rank = 0
        self.world_size = 1
        self.expected_epoch_iterations = default_length
        self.total_length = -1
        self.worker_id = 0
        self.number_of_workers = 1
        self._update_slice()

    def _update_slice(self):
        self.slice_index = self.rank * self.world_size + self.worker_id
        self.slice_length = self.world_size * self.number_of_workers
        self.epoch_iterations = self.expected_epoch_iterations // self.slice_length * self.slice_length

    def set_rank(self, rank, world_size):
        self.rank = rank
        self.world_size = world_size
        self._update_slice()

    def set_size(self, epoch_iterations, total_length):
        self.expected_epoch_iterations = epoch_iterations
        self.total_length = total_length
        self._update_slice()

    def set_worker_id(self, worker_id, number_of_workers):
        self.worker_id = worker_id
        self.number_of_workers = number_of_workers
        self._update_slice()

    def get_action(self, index):
        if 0 < self.total_length <= index:
            return SliceManager.Action.Stop

        if index % self.slice_length != self.slice_index:
            return SliceManager.Action.Skip

        return SliceManager.Action.Do


class TrackingDataset(torch.utils.data.dataset.IterableDataset):
    def __init__(self, datasets_sampler, decoding_phase_processor, data_preprocessing_phase_processor, epoch_iterations):
        self.datasets_sampler = datasets_sampler
        self.decoding_phase_processor = decoding_phase_processor
        self.data_preprocessing_phase_processor = data_preprocessing_phase_processor
        self.running_datasets_sampler = None
        self.slice_manager = SliceManager(epoch_iterations)
        self.index = 0

    def __iter__(self):
        if self.running_datasets_sampler is None:
            self.running_datasets_sampler = copy.deepcopy(self.datasets_sampler)
        self.epoch_iteration_count = 0
        return self

    def __next__(self):
        while True:
            if self.epoch_iteration_count >= self.slice_manager.epoch_iterations:
                raise StopIteration

            action = self.slice_manager.get_action(self.index)
            if action == SliceManager.Action.Stop:
                self.running_datasets_sampler = copy.deepcopy(self.datasets_sampler)
                self.index = 0
                continue

            sample = self.running_datasets_sampler.get_next()

            self.index += 1
            self.epoch_iteration_count += 1

            if action == SliceManager.Action.Do:
                result = self.data_preprocessing_phase_processor(*self.decoding_phase_processor(*sample))
                return result

    def get_slice_manager(self):
        return self.slice_manager


def build_tracking_data_loader(train_config: dict, dataset_config: dict):

    pass
