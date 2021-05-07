from Dataset.DET.Storage.MemoryMapped.dataset import DetectionDatasetImage_MemoryMapped
import numpy as np


class DetectionDatasetImageObjectSampler:
    def __init__(self, data_getter, rng_engine, sampling_allow_invalid_bounding_box=False):
        self.data_getter = data_getter
        self.rng_engine = rng_engine
        self.sampling_allow_invalid_bounding_box = sampling_allow_invalid_bounding_box

    def __call__(self, image: DetectionDatasetImage_MemoryMapped):
        number_of_objects = len(image)
        if not self.sampling_allow_invalid_bounding_box and image.has_bounding_box_validity_flag():
            indices = np.arange(0, number_of_objects)
            indices = indices[image.get_all_bounding_box_validity_flag()]
            index = self.rng_engine.choice(indices)
        else:
            index = self.rng_engine.randint(0, number_of_objects)
        return [self.data_getter(image, image[index])]
