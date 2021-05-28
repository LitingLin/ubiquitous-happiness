from Dataset.SOT.Storage.MemoryMapped.dataset import SingleObjectTrackingDatasetSequence_MemoryMapped
import numpy as np
from data.tracking.sampler.sampling.dataset_samplers._sampling_algos.stateless.random import sampling_multiple_indices_with_range_and_mask


def do_positive_sampling_in_single_object_tracking_sequence(sequence: SingleObjectTrackingDatasetSequence_MemoryMapped):
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


def do_negative_sampling_in_single_object_tracking_sequence():
    pass


def do_sampling_in_single_object_tracking_sequence():
    pass
