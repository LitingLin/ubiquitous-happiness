from Dataset.MOT.Storage.MemoryMapped.dataset import MultipleObjectTrackingDataset_MemoryMapped, MultipleObjectTrackingDatasetSequence_MemoryMapped, MultipleObjectTrackingDatasetSequenceObject_MemoryMapped
import numpy as np


class _MOTTrackIterator:
    def __init__(self, sequence: MultipleObjectTrackingDatasetSequence_MemoryMapped, sequence_object: MultipleObjectTrackingDatasetSequenceObject_MemoryMapped):
        self.sequence = sequence
        self.sequence_object = sequence_object
        self.frame_indices = sequence_object.get_all_frame_index()
        self.frame_index_iter = iter(range(min(self.frame_indices), max(self.frame_indices) + 1))

    def __next__(self):
        frame_index = next(self.frame_index_iter)
        frame = self.sequence[frame_index]
        if frame_index not in self.frame_indices:
            return frame.get_image_path(), None, False
        object_ = frame.get_object_by_id(self.sequence_object.get_id())
        bounding_box = object_.get_bounding_box()
        if object_.has_bounding_box_validity_flag():
            validity_flag = object_.get_bounding_box_validity_flag()
        else:
            validity_flag = True
        return frame.get_image_path(), bounding_box, validity_flag


class _MOTTrackIteratorGenerator:
    def __init__(self, sequence: MultipleObjectTrackingDatasetSequence_MemoryMapped, sequence_object: MultipleObjectTrackingDatasetSequenceObject_MemoryMapped):
        assert sequence.has_bounding_box()
        assert len(sequence_object.get_all_frame_index()) > 0
        self.sequence = sequence
        self.sequence_object = sequence_object

    def __iter__(self):
        return _MOTTrackIterator(self.sequence, self.sequence_object)


class _MOTWrapper:
    def __init__(self, dataset: MultipleObjectTrackingDataset_MemoryMapped):
        self.dataset = dataset

    def get_random_track(self):
        index_of_sequence = np.random.randint(0, len(self.dataset))
        return self.get_random_track_in_sequence(index_of_sequence)

    def get_random_track_in_sequence(self, index: int):
        sequence = self.dataset[index]
        number_of_objects = sequence.get_number_of_objects()
        assert number_of_objects > 0
        index_of_object_in_sequence = np.random.randint(0, number_of_objects)
        sequence_object = sequence.get_object(index_of_object_in_sequence)
        return _MOTTrackIteratorGenerator(sequence, sequence_object)

    def get_track_in_sequence(self, index_of_sequence: int, index_of_track_in_sequence: int):
        sequence = self.dataset[index_of_sequence]
        sequence_object = sequence.get_object(index_of_track_in_sequence)
        return _MOTTrackIteratorGenerator(sequence, sequence_object)

    def get_number_of_tracks_in_sequence(self, index_of_sequence: int):
        sequence = self.dataset[index_of_sequence]
        return sequence.get_number_of_objects()

    def __len__(self):
        return len(self.dataset)
