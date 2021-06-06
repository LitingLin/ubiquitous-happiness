from Dataset.MOT.Storage.MemoryMapped.dataset import MultipleObjectTrackingDatasetSequence_MemoryMapped
from ._algo import do_siamfc_pair_sampling_positive_only, do_siamfc_pair_sampling_negative_only, do_siamfc_pair_sampling
import numpy as np
from ._dummy_bbox import generate_dummy_bbox_xyxy
from data.operator.bbox.validity import bbox_is_valid
from data.operator.bbox.spatial.utility.aligned.image import bounding_box_is_intersect_with_image


def _data_getter(sequence: MultipleObjectTrackingDatasetSequence_MemoryMapped, track_id, index_of_frames, rng_engine: np.random.Generator):
    z = sequence.get_frame(index_of_frames[0])
    z_image = z.get_image_path()
    z_bbox = z.get_object_by_id(track_id).get_bounding_box()

    assert any(v > 0 for v in z.get_image_size())
    assert bbox_is_valid(z_bbox) and bounding_box_is_intersect_with_image(z_bbox, z.get_image_size())

    if len(index_of_frames) == 1:
        return ((z_image, z_bbox), )

    x = sequence.get_frame(index_of_frames[1])
    x_image = x.get_image_path()
    if x.has_object(track_id):
        x_obj_info = x.get_object_by_id(track_id)
        x_bbox_validity = x_obj_info.get_bounding_box_validity_flag()
        if x_bbox_validity is not None and not x_bbox_validity:
            x_bbox = generate_dummy_bbox_xyxy(x.get_image_size(), rng_engine, z_bbox)
        else:
            x_bbox = x_obj_info.get_bounding_box()
    else:
        x_bbox = generate_dummy_bbox_xyxy(x.get_image_size(), rng_engine, z_bbox)
    assert any(v > 0 for v in x.get_image_size())
    assert bbox_is_valid(x_bbox) and bounding_box_is_intersect_with_image(x_bbox, x.get_image_size())

    return ((z_image, z_bbox), (x_image, x_bbox))


def _sampling_one_track_in_sequence_and_generate_object_visible_mask(sequence: MultipleObjectTrackingDatasetSequence_MemoryMapped, rng_engine: np.random.Generator):
    index_of_track = rng_engine.integers(0, sequence.get_number_of_objects())
    track = sequence.get_object(index_of_track)

    mask = np.zeros(len(sequence), dtype=np.bool_)
    if track.get_all_bounding_box_validity_flag() is not None:
        ind = track.get_all_frame_index()[track.get_all_bounding_box_validity_flag()]
        mask[ind] = True
    else:
        mask[track.get_all_frame_index()] = True
    return mask, track.get_id()


def do_positive_sampling_in_multiple_object_tracking_dataset_sequence(sequence: MultipleObjectTrackingDatasetSequence_MemoryMapped, frame_range: int, rng_engine: np.random.Generator):
    mask, track_id = _sampling_one_track_in_sequence_and_generate_object_visible_mask(sequence, rng_engine)

    return _data_getter(sequence, track_id, do_siamfc_pair_sampling_positive_only(len(sequence), frame_range, mask, rng_engine), rng_engine)


def do_negative_sampling_in_multiple_object_tracking_dataset_sequence(sequence: MultipleObjectTrackingDatasetSequence_MemoryMapped, frame_range: int, rng_engine: np.random.Generator):
    mask, track_id = _sampling_one_track_in_sequence_and_generate_object_visible_mask(sequence, rng_engine)

    return _data_getter(sequence, track_id, do_siamfc_pair_sampling_negative_only(len(sequence), frame_range, mask, rng_engine), rng_engine)


def do_sampling_in_multiple_object_tracking_dataset_sequence(sequence: MultipleObjectTrackingDatasetSequence_MemoryMapped, frame_range: int, rng_engine: np.random.Generator):
    mask, track_id = _sampling_one_track_in_sequence_and_generate_object_visible_mask(sequence, rng_engine)

    indices, is_positive = do_siamfc_pair_sampling(len(sequence), frame_range, mask, rng_engine)
    return _data_getter(sequence, track_id, indices, rng_engine), is_positive


def get_one_random_sample_in_multiple_object_tracking_dataset_sequence(sequence: MultipleObjectTrackingDatasetSequence_MemoryMapped, rng_engine: np.random.Generator):
    index_of_frame = rng_engine.integers(0, sequence.get_number_of_frames())
    frame = sequence[index_of_frame]
    if len(frame) == 0:
        return frame.get_image_path(), generate_dummy_bbox_xyxy(frame.get_image_size(), rng_engine)
    else:
        index_of_frame_object = rng_engine.integers(0, len(frame))
        frame_object = frame[index_of_frame_object]
        bbox_validity = frame_object.get_bounding_box_validity_flag()
        if bbox_validity is not None and not bbox_validity:
            return frame.get_image_path(), generate_dummy_bbox_xyxy(frame.get_image_size(), rng_engine)
        else:
            return frame.get_image_path(), frame_object.get_bounding_box()
