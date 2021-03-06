from Dataset.MOT.Storage.MemoryMapped.dataset import MultipleObjectTrackingDataset_MemoryMapped
import numpy as np


class MultipleObjectTrackingDatasetSampler:
    def __init__(self, dataset: MultipleObjectTrackingDataset_MemoryMapped, sequence_sampler):
        self.dataset = dataset
        self.sequence_sampler = sequence_sampler

    def get_random_track(self):
        index_of_sequence = np.random.randint(0, len(self.dataset))
        return self.get_random_track_in_sequence(index_of_sequence)

    @staticmethod
    def _check_sequence(sequence, sequence_object):
        assert sequence.has_bounding_box()
        assert len(sequence_object.get_all_frame_index()) > 0

    def get_random_track_in_sequence(self, index: int):
        sequence = self.dataset[index]
        number_of_objects = sequence.get_number_of_objects()
        assert number_of_objects > 0
        index_of_object_in_sequence = np.random.randint(0, number_of_objects)
        sequence_object = sequence.get_object(index_of_object_in_sequence)
        MultipleObjectTrackingDatasetSampler._check_sequence(sequence, sequence_object)
        return self.sequence_sampler(sequence, sequence_object)

    def get_track_in_sequence(self, index_of_sequence: int, index_of_track_in_sequence: int):
        sequence = self.dataset[index_of_sequence]
        sequence_object = sequence.get_object(index_of_track_in_sequence)
        MultipleObjectTrackingDatasetSampler._check_sequence(sequence, sequence_object)
        return self.sequence_sampler(sequence, sequence_object)

    def get_number_of_tracks_in_sequence(self, index_of_sequence: int):
        sequence = self.dataset[index_of_sequence]
        return sequence.get_number_of_objects()

    def __len__(self):
        return len(self.dataset)
