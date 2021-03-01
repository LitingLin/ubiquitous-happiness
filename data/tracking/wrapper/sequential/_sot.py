from Dataset.SOT.Storage.MemoryMapped.dataset import SingleObjectTrackingDataset_MemoryMapped, \
    SingleObjectTrackingDatasetSequence_MemoryMapped, SingleObjectTrackingDatasetFrame_MemoryMapped
import numpy as np


class _SOTTrackIterator:
    def __init__(self, sequence: SingleObjectTrackingDatasetSequence_MemoryMapped):
        self.sequence = sequence
        self.sequence_iter = iter(sequence)

    def __next__(self):
        frame: SingleObjectTrackingDatasetFrame_MemoryMapped = next(self.sequence_iter)
        image_path = frame.get_image_path()
        bounding_box = frame.get_bounding_box()
        if frame.has_bounding_box_validity_flag():
            validity_flag = frame.get_bounding_box_validity_flag()
        else:
            validity_flag = True
        return image_path, bounding_box, validity_flag


class _SOTTrackIteratorGenerator:
    def __init__(self, sequence: SingleObjectTrackingDatasetSequence_MemoryMapped):
        assert sequence.has_bounding_box()
        assert len(sequence) > 0
        self.sequence = sequence

    def __iter__(self):
        return _SOTTrackIterator(self.sequence)


class _SOTWrapper:
    def __init__(self, dataset: SingleObjectTrackingDataset_MemoryMapped):
        self.dataset = dataset

    def get_random_track(self):
        index = np.random.randint(0, len(self.dataset))
        return self.get_random_track_in_sequence(index)

    def get_random_track_in_sequence(self, index: int):
        return _SOTTrackIteratorGenerator(self.dataset[index])

    def get_track_in_sequence(self, index_of_sequence: int, index_of_track: int):
        assert index_of_track == 0
        return _SOTTrackIteratorGenerator(self.dataset[index_of_sequence])

    def get_number_of_tracks_in_sequence(self, index_of_sequence: int):
        return 1

    def __len__(self):
        return len(self.dataset)
