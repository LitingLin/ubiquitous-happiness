from Dataset.SOT.Storage.MemoryMapped.dataset import SingleObjectTrackingDataset_MemoryMapped
import numpy as np


class SingleObjectTrackingDatasetSampler:
    def __init__(self, dataset: SingleObjectTrackingDataset_MemoryMapped, sequence_sampler):
        self.dataset = dataset
        self.sequence_sampler = sequence_sampler

    def get_random_track(self):
        index = np.random.randint(0, len(self.dataset))
        return self.get_random_track_in_sequence(index)

    def get_random_track_in_sequence(self, index: int):
        return self.sequence_sampler(self.dataset[index])

    def get_track_in_sequence(self, index_of_sequence: int, index_of_track: int):
        assert index_of_track == 0
        return self.sequence_sampler(self.dataset[index_of_sequence])

    def get_number_of_tracks_in_sequence(self, index_of_sequence: int):
        return 1

    def __len__(self):
        return len(self.dataset)
