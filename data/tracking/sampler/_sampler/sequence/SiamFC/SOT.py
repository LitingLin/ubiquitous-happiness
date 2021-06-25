from Dataset.SOT.Storage.MemoryMapped.dataset import SingleObjectTrackingDatasetSequence_MemoryMapped
from ._algo import do_siamfc_pair_sampling_positive_only, do_siamfc_pair_sampling_negative_only, do_siamfc_pair_sampling
from data.tracking.sampler._sampler.sequence.common._dummy_bbox import generate_dummy_bbox_xyxy
import numpy as np
from data.operator.bbox.validity import bbox_is_valid
from data.operator.bbox.spatial.utility.aligned.image import bounding_box_is_intersect_with_image
from data.tracking.sampler.SiamFC.type import SiamesePairSamplingMethod


def _data_getter(sequence, indices, rng_engine: np.random.Generator):
    z = sequence[indices[0]]
    z_image = z.get_image_path()
    z_bbox = z.get_bounding_box()
    assert any(v > 0 for v in z.get_image_size())
    assert bbox_is_valid(z_bbox) and bounding_box_is_intersect_with_image(z_bbox, z.get_image_size())
    if len(indices) == 1:
        return ((z_image, z_bbox), )

    x = sequence[indices[1]]
    x_image = x.get_image_path()
    x_bbox = x.get_bounding_box()
    x_bbox_validity = x.get_bounding_box_validity_flag()
    if x_bbox_validity is not None and not x_bbox_validity:
        x_bbox = generate_dummy_bbox_xyxy(x.get_image_size(), rng_engine, z_bbox)
    else:
        assert bbox_is_valid(x_bbox) and bounding_box_is_intersect_with_image(x_bbox, x.get_image_size())
    assert any(v > 0 for v in x.get_image_size())
    return ((z_image, z_bbox), (x_image, x_bbox))


def do_positive_sampling_in_single_object_tracking_dataset_sequence(sequence: SingleObjectTrackingDatasetSequence_MemoryMapped, frame_range: int, sampling_method: SiamesePairSamplingMethod, rng_engine: np.random.Generator):
    return _data_getter(sequence, do_siamfc_pair_sampling_positive_only(len(sequence), frame_range, sequence.get_all_bounding_box_validity_flag(), sampling_method, rng_engine), rng_engine)


def do_negative_sampling_in_single_object_tracking_dataset_sequence(sequence: SingleObjectTrackingDatasetSequence_MemoryMapped, frame_range: int, rng_engine: np.random.Generator):
    return _data_getter(sequence, do_siamfc_pair_sampling_negative_only(len(sequence), frame_range, sequence.get_all_bounding_box_validity_flag(), rng_engine), rng_engine)


def do_sampling_in_single_object_tracking_dataset_sequence(sequence: SingleObjectTrackingDatasetSequence_MemoryMapped, frame_range: int, sampling_method: SiamesePairSamplingMethod, rng_engine: np.random.Generator):
    indices, is_positive = do_siamfc_pair_sampling(len(sequence), frame_range, sequence.get_all_bounding_box_validity_flag(), sampling_method, rng_engine)
    return _data_getter(sequence, indices, rng_engine), is_positive
