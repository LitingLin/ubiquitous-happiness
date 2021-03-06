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


class SingleObjectTrackingDatasetTrackIteratorGenerator:
    def __init__(self, sequence: SingleObjectTrackingDatasetSequence_MemoryMapped):
        self.sequence = sequence

    def __iter__(self):
        return _SOTTrackIterator(self.sequence)

    def __len__(self):
        return len(self.sequence)
