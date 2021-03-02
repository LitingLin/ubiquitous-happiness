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


class MultipleObjectTrackingDatasetTrackIteratorGenerator:
    def __init__(self, sequence: MultipleObjectTrackingDatasetSequence_MemoryMapped, sequence_object: MultipleObjectTrackingDatasetSequenceObject_MemoryMapped):
        assert sequence.has_bounding_box()
        assert len(sequence_object.get_all_frame_index()) > 0
        self.sequence = sequence
        self.sequence_object = sequence_object

    def __iter__(self):
        return _MOTTrackIterator(self.sequence, self.sequence_object)
