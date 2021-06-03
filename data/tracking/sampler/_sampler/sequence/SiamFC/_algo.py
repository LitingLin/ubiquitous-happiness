from data.tracking.sampler._sampling_algos.stateless.random import sampling_multiple_indices_with_range_and_mask, sampling, sampling_with_mask
import numpy as np


def sample_one_positive(length, mask, rng_engine):
    if mask is None:
        z_index = sampling(length, rng_engine)
    else:
        z_index = sampling_with_mask(mask, rng_engine)

    return z_index


def do_siamfc_pair_sampling(length: int, frame_range: int, mask: np.ndarray=None, rng_engine=np.random):
    z_index = sample_one_positive(length, mask, rng_engine)

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
    z_index = sample_one_positive(length, mask, rng_engine)
    if mask is None or length == 1:
        return (z_index,)

    begin = z_index - frame_range
    end = z_index + frame_range
    x_axis_begin_value = -begin * 8 / (2 * frame_range + 1) - 4
    x_axis_end_value = (length - 1 - end) * 8 / (2 * frame_range + 1) + 4
    x_axis_values = np.linspace(x_axis_begin_value, x_axis_end_value, length)
    not_mask = ~mask
    not_mask[z_index] = False
    x_axis_values = x_axis_values[not_mask]
    if len(x_axis_values) == 0:
        return (z_index,)
    probability = _gaussian(x_axis_values, 0., 5.)
    probability = probability / probability.sum()
    candidates = np.arange(0, length)[not_mask]
    x_index = rng_engine.choice(candidates, p=probability)
    return z_index, x_index


if __name__ == '__main__':
    a = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
       1, 1, 1, 1, 1, 1, 1], dtype=np.bool_)

    do_siamfc_pair_sampling(29, 3, a, np.random.default_rng())

