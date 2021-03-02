import numpy as np


def _arg_sample(length, validity_flags: np.ndarray=None, number_of_objects=1, sampling_range_size=None, allow_duplication=True, allow_insufficiency=True, sort=False, rng_engine=np.random):
    assert number_of_objects > 0
    if validity_flags is not None:
        assert length == len(validity_flags)
    if sampling_range_size is not None and sampling_range_size < length:
        range_begin_index = rng_engine.randint(0, length - sampling_range_size + 1)
        range_end_index = range_begin_index + sampling_range_size
        sampling_indices = np.arange(range_begin_index, range_end_index)
        if validity_flags is not None:
            sampling_indices = sampling_indices[validity_flags[range_begin_index: range_end_index]]
    else:
        sampling_indices = np.arange(0, length)
        if validity_flags is not None:
            sampling_indices = sampling_indices[validity_flags]
    length_of_sampling_indices = sampling_indices.shape[0]
    if length_of_sampling_indices < number_of_objects:
        if length_of_sampling_indices == 0:
            return None
        if not allow_insufficiency:
            return None
        number_of_objects = length_of_sampling_indices

    indices = rng_engine.choice(sampling_indices, number_of_objects, replace=allow_duplication)
    if sort:
        indices = np.sort(indices)
    return indices
