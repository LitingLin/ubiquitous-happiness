from Dataset.SOT.Storage.MemoryMapped.dataset import SingleObjectTrackingDatasetSequence_MemoryMapped
from ._sampling import _arg_sample
import numpy as np


class SingleObjectTrackingDatasetSequenceSampler:
    def __init__(self, data_getter, number_of_objects=1, frame_range_size=None, sampling_allow_duplication=True,
                 sampling_allow_insufficiency=True, sort_result=False, rng_engine=np.random):
        self.data_getter = data_getter
        self.frame_range_size = frame_range_size
        self.number_of_objects = number_of_objects
        self.sampling_allow_duplication = sampling_allow_duplication
        self.sampling_allow_insufficiency = sampling_allow_insufficiency
        self.sort_result = sort_result
        self.rng_engine = rng_engine

    def _get_data_from_indices(self, sequence: SingleObjectTrackingDatasetSequence_MemoryMapped, indices):
        if indices is None:
            return None
        data = []
        for index in indices:
            frame = sequence[index]
            data.append(self.data_getter(sequence, frame))
        return data

    def __call__(self, sequence: SingleObjectTrackingDatasetSequence_MemoryMapped):
        if sequence.has_bounding_box_validity_flag():
            validity_flags_vector = sequence.get_all_bounding_box_validity_flag()
        else:
            validity_flags_vector = None
        return self._get_data_from_indices(sequence,
                                           _arg_sample(len(sequence), validity_flags_vector, self.number_of_objects,
                                                       self.frame_range_size,
                                                       self.sampling_allow_duplication,
                                                       self.sampling_allow_insufficiency, self.sort_result,
                                                       self.rng_engine))
