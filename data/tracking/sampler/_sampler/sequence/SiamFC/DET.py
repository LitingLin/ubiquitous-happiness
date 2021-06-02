import numpy as np
from Dataset.DET.Storage.MemoryMapped.dataset import DetectionDatasetImage_MemoryMapped
from ._algo import sample_one_positive, sampling
from ._dummy_bbox import generate_dummy_bbox_xyxy


def do_sampling_in_detection_dataset_image(image: DetectionDatasetImage_MemoryMapped, rng_engine: np.random.Generator):
    index_of_object = sample_one_positive(len(image), image.get_all_bounding_box_validity_flag(), rng_engine)
    object_ = image[index_of_object]
    return image.get_image_path(), object_.get_bounding_box()


def get_one_random_sample_in_detection_dataset_image(image: DetectionDatasetImage_MemoryMapped, rng_engine: np.random.Generator):
    index_of_object = sampling(len(image), rng_engine)
    object_ = image[index_of_object]
    if object_.get_bounding_box_validity_flag() is False:
        bbox = generate_dummy_bbox_xyxy(image.get_image_size(), rng_engine)
    else:
        bbox = object_.get_bounding_box()
    return image.get_image_path(), bbox
