from Dataset.SOT.Storage.MemoryMapped.dataset import SingleObjectTrackingDatasetSequence_MemoryMapped
from ._algo import do_siamfc_pair_sampling_positive_only, do_siamfc_pair_sampling_negative_only, do_siamfc_pair_sampling
from ._dummy_bbox import generate_dummy_bbox_xyxy


def _data_getter(sequence, indices, rng_engine):
    z = sequence[indices[0]]
    z_image = z.get_image_path()
    z_bbox = z.get_bounding_box()
    if len(indices) == 1:
        return ((z_image, z_bbox), )

    x = sequence[indices[1]]
    x_image = x.get_image_path()
    x_bbox = x.get_bounding_box()
    x_bbox_validity = x.get_bounding_box_validity_flag()
    if x_bbox_validity is False:
        x_bbox = generate_dummy_bbox_xyxy(x.get_image_size(), rng_engine, z_bbox)
    return ((z_image, z_bbox), (x_image, x_bbox))


def do_positive_sampling_in_single_object_tracking_dataset_sequence(sequence: SingleObjectTrackingDatasetSequence_MemoryMapped, frame_range: int, rng_engine):
    return _data_getter(sequence, do_siamfc_pair_sampling_positive_only(len(sequence), frame_range, sequence.get_all_bounding_box_validity_flag(), rng_engine), rng_engine)


def do_negative_sampling_in_single_object_tracking_dataset_sequence(sequence: SingleObjectTrackingDatasetSequence_MemoryMapped, frame_range: int, rng_engine):
    return _data_getter(sequence, do_siamfc_pair_sampling_negative_only(len(sequence), frame_range, sequence.get_all_bounding_box_validity_flag(), rng_engine), rng_engine)


def do_sampling_in_single_object_tracking_dataset_sequence(sequence: SingleObjectTrackingDatasetSequence_MemoryMapped, frame_range: int, rng_engine):
    indices, is_positive = do_siamfc_pair_sampling(len(sequence), frame_range, sequence.get_all_bounding_box_validity_flag(), rng_engine)
    return _data_getter(sequence, indices, rng_engine), is_positive


def get_one_random_sample_in_single_object_tracking_dataset_sequence(sequence: SingleObjectTrackingDatasetSequence_MemoryMapped, rng_engine):
    index_of_frame = rng_engine.randint(0, len(sequence))
    frame = sequence[index_of_frame]
    if frame.get_bounding_box_validity_flag() is False:
        return frame.get_image_path(), generate_dummy_bbox_xyxy(frame.get_image_size(), rng_engine)
    else:
        return frame.get_image_path(), frame.get_bounding_box()
