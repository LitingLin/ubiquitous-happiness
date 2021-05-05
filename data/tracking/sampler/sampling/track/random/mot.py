from Dataset.MOT.Storage.MemoryMapped.dataset import MultipleObjectTrackingDatasetSequence_MemoryMapped, MultipleObjectTrackingDatasetSequenceObject_MemoryMapped
from ._sampling import _arg_sample
import numpy as np


class MultipleObjectTrackingDatasetSequenceSampler:
    def __init__(self, data_getter, number_of_objects = 1, frame_range_size=None, sampling_allow_duplication=True, sampling_allow_insufficiency=True, sort_result=False, rng_engine=np.random):
        self.data_getter = data_getter
        self.frame_range_size = frame_range_size
        self.number_of_objects = number_of_objects
        self.sampling_allow_duplication = sampling_allow_duplication
        self.sampling_allow_insufficiency = sampling_allow_insufficiency
        self.sort_result = sort_result
        self.rng_engine = rng_engine

    def _get_data_from_indices(self, sequence, sequence_object, indices, offset):
        if indices is None:
            return None
        data = []
        for index in indices:
            frame = sequence[index + offset]
            frame_object = frame.get_object_by_id(sequence_object.get_id())
            data.append(self.data_getter(sequence, frame, sequence_object, frame_object))
        return data

    def __call__(self, sequence: MultipleObjectTrackingDatasetSequence_MemoryMapped, sequence_object: MultipleObjectTrackingDatasetSequenceObject_MemoryMapped):
        frame_indices = sequence_object.get_all_frame_index()
        track_frame_index_offset = min(frame_indices)
        track_length = max(frame_indices) - track_frame_index_offset + 1
        validity_flags_vector = np.zeros(track_length, dtype=np.bool)
        frame_indices = frame_indices - track_frame_index_offset
        if sequence.has_bounding_box_validity_flag():
            frame_indices = frame_indices[sequence_object.get_all_bounding_box_validity_flag()]
        validity_flags_vector[frame_indices] = True

        indices = _arg_sample(track_length, validity_flags_vector, self.number_of_objects, self.frame_range_size, self.sampling_allow_duplication, self.sampling_allow_insufficiency, self.sort_result, self.rng_engine)
        return self._get_data_from_indices(sequence, sequence_object, indices, track_frame_index_offset)
