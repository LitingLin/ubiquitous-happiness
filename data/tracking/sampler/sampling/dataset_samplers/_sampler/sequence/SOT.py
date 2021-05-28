from Dataset.SOT.Storage.MemoryMapped.dataset import SingleObjectTrackingDatasetSequence_MemoryMapped
import numpy as np
from data.tracking.sampler.sampling.dataset_samplers._sampling_algos.stateless.random import sampling_multiple_indices_with_range_and_mask, sampling, sampling_with_mask


def do_positive_sampling_in_single_object_tracking_sequence(sequence: SingleObjectTrackingDatasetSequence_MemoryMapped, frame_range: int):
    visible = sequence.get_all_bounding_box_validity_flag()
    if visible is None:
        visible = np.ones([len(sequence)], dtype=np.uint8)
    z_index = _sample_visible_ids(visible)
    if z_index is None:
        return None

    z_index = z_index[0]
    z_frame = sequence[z_index]

    x_index = _sample_visible_ids(visible, 1, z_index - frame_range, z_index + frame_range)
    if x_index is None or x_index[0] == z_index:
        return z_frame.get_image_path(), z_frame.get_bounding_box()
    else:
        x_frame = sequence[x_index[0]]
        return z_frame.get_image_path(), z_frame.get_bounding_box(), x_frame.get_image_path(), x_frame.get_bounding_box()

    pass


def _sample_one_positive(length, mask, rng_engine):
    if mask is None:
        z_index = sampling(length, rng_engine)
    else:
        z_index = sampling_with_mask(mask, rng_engine)

    return z_index


def do_negative_sampling_in_single_object_tracking_sequence(sequence: SingleObjectTrackingDatasetSequence_MemoryMapped, frame_range: int, rng_engine=np.random):
    bounding_box_validity_mask = sequence.get_all_bounding_box_validity_flag()
    z_index = _sample_one_positive_frame(len(sequence), bounding_box_validity_mask, rng_engine)
    _sample_one_positive_frame(len(sequence), )


def _data_getter(frames):
    return tuple((frame.get_image_path(), frame.get_bounding_box()) for frame in frames)


def do_siamfc_pair_sampling(length: int, frame_range: int, mask: np.ndarray=None, rng_engine=np.random):
    z_index = _sample_one_positive(length, mask, rng_engine)

    if length == 1:
        return (z_index,), 0

    x_frame_begin = z_index - frame_range
    x_frame_begin = max(x_frame_begin, 0)
    x_frame_end = z_index + frame_range + 1
    x_frame_end = min(x_frame_end, length)

    x_candidate_indices = np.arange(x_frame_begin, x_frame_end)
    if mask is None:
        x_candidate_indices = np.delete(x_candidate_indices, z_index - x_frame_begin)
    else:
        x_candidate_indices_mask = np.copy(mask[x_frame_begin: x_frame_end])
        x_candidate_indices_mask[z_index - x_frame_begin] = False
        x_candidate_indices = x_candidate_indices[x_candidate_indices_mask]
        if len(x_candidate_indices) == 0:
            return (z_index,), 0

    x_index = rng_engine.choice(x_candidate_indices)
    if mask is not None and not mask[x_index]:
        is_positive = -1
    else:
        is_positive = 1
    return (z_index, x_index), is_positive


def do_siamfc_pair_sampling_positive_only(length: int, frame_range: int, mask: np.ndarray=None, rng_engine=np.random):
    return sampling_multiple_indices_with_range_and_mask(length, mask, 2, frame_range, allow_duplication=False, allow_insufficiency=True, sort=False, rng_engine=rng_engine)


def _gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


def do_siamfc_pair_sampling_negative_only(length: int, frame_range: int, mask: np.ndarray=None, rng_engine=np.random):
    z_index = _sample_one_positive(length, mask, rng_engine)
    if mask is None or length == 1:
        return z_index

    begin = z_index - frame_range
    end = z_index + frame_range
    x_axis_begin_value = -begin * 8 / (2 * frame_range + 1) - 4
    x_axis_end_value = (length - 1 - end) * 8 / (2 * frame_range + 1) + 4
    x_axis_values = np.linspace(x_axis_begin_value, x_axis_end_value, length)
    not_mask = ~mask
    not_mask[z_index] = False
    x_axis_values = x_axis_values[not_mask]
    if len(x_axis_values) == 0:
        return z_index
    probability = _gaussian(x_axis_values, 0., 5.)
    probability = probability / probability.sum()
    candidates = np.arange(0, length)[not_mask]
    x_index = rng_engine.choice(candidates, p=probability)
    return z_index, x_index

def do_sampling_in_single_object_tracking_sequence(sequence: SingleObjectTrackingDatasetSequence_MemoryMapped, frame_range: int, rng_engine=np.random):
    bounding_box_validity_mask = sequence.get_all_bounding_box_validity_flag()
    z_index = _sample_one_positive_frame(len(sequence), bounding_box_validity_mask, rng_engine)
    if bounding_box_validity_mask is None:
        z_index = sampling(len(sequence), rng_engine)
    else:
        z_index = sampling_with_mask(bounding_box_validity_mask, rng_engine)

    z_frame = sequence[z_index]

    if len(sequence) == 1:
        return _data_getter((z_frame,)), 0

    x_frame_begin = z_frame - frame_range
    x_frame_begin = max(x_frame_begin, 0)
    x_frame_end = z_frame + frame_range + 1
    x_frame_end = min(x_frame_end, len(sequence))

    x_candidate_indices = np.arange(x_frame_begin, x_frame_end)
    if bounding_box_validity_mask is None:
        x_candidate_indices = np.delete(x_candidate_indices, z_index - x_frame_begin)
    else:
        x_candidate_indices_mask = np.copy(bounding_box_validity_mask[x_frame_begin: x_frame_end])
        x_candidate_indices_mask[z_index - x_frame_begin] = False
        x_candidate_indices = x_candidate_indices[x_candidate_indices_mask]
        if len(x_candidate_indices) == 0:
            return _data_getter((z_frame,)), 0

    x_index = rng_engine.choice(x_candidate_indices)
    x_frame = sequence[x_index]
    is_positive = 1
    if bounding_box_validity_mask is not None and not x_frame.get_bounding_box_validity_flag():
        is_positive = -1

    return _data_getter((z_frame, x_frame)), is_positive
