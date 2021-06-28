import numpy as np
from data.tracking.sampler._sampler.sequence.SiamFC._algo import do_siamfc_pair_sampling


def do_trident_sampling(length: int, frame_range: int, mask: np.ndarray=None, rng_engine: np.random.Generator=np.random.default_rng()):
    do_siamfc_pair_sampling(length, frame_range, mask, rng_engine)


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


def do_trident_positive_sampling(length: int, frame_range: int, mask: np.ndarray=None, rng_engine: np.random.Generator=np.random.default_rng()):
    pass

def do_trident_negative_sampling(length: int, frame_range: int, mask: np.ndarray=None, rng_engine: np.random.Generator=np.random.default_rng()):
    pass
