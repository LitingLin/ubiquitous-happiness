from data.tracking.sampler.sampling._impl.random.dataset.sample_sequence_without_replacement_and_sample_subtrack_fully_randomly.det import DetectionDatasetSampler
from data.tracking.sampler.sampling._impl.random.track.random.det import DetectionDatasetImageObjectSampler


class DETDatasetRandomSampler:
    def __init__(self, dataset, data_getter, rng_engine, sampling_allow_invalid_bounding_box=False):
        self.dataset_sampler = DetectionDatasetSampler(dataset, rng_engine)
        self.image_object_sampler = DetectionDatasetImageObjectSampler(data_getter, rng_engine, sampling_allow_invalid_bounding_box)

    def get_next(self):
        return self.image_object_sampler(self.dataset_sampler.get_next())
