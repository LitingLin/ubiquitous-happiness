import numpy as np


# may return None when allow_insufficiency=False
def _arg_sample(length, validity_flags: np.ndarray=None, number_of_objects=1, sampling_range_size=None, allow_duplication=True, allow_insufficiency=True, sort=False, rng_engine=np.random):
    assert number_of_objects > 0
    if validity_flags is not None:
        assert length == len(validity_flags)

    if sampling_range_size is not None and sampling_range_size < length:
        if validity_flags is not None:
            sampling_indices = np.arange(0, length)
            sampling_indices = sampling_indices[validity_flags]
        else:
            sampling_indices = length
        range_mid_index = rng_engine.choice(sampling_indices)
        range_index = [range_mid_index - sampling_range_size // 2, range_mid_index + (sampling_range_size - sampling_range_size // 2)]
        range_index[0] = max(range_index[0], 0)
        range_index[1] = min(range_index[1], length)
        sampling_indices = np.arange(*range_index)
        if validity_flags is not None:
            sampling_indices = sampling_indices[validity_flags[range_index[0]: range_index[1]]]
    else:
        sampling_indices = np.arange(0, length)
        if validity_flags is not None:
            sampling_indices = sampling_indices[validity_flags]
    length_of_sampling_indices = sampling_indices.shape[0]
    if not allow_duplication:
        if length_of_sampling_indices < number_of_objects:
            if not allow_insufficiency:
                return None
            number_of_objects = length_of_sampling_indices

    indices = rng_engine.choice(sampling_indices, number_of_objects, replace=allow_duplication)
    if sort:
        indices = np.sort(indices)
    return indices
