from data.tracking.sampler.sampling._impl.random.dataset.without_replacement.sot import SingleObjectTrackingDatasetSampler
from data.tracking.sampler.sampling._impl.random.track.random.sot import SingleObjectTrackingDatasetSequenceSampler


class SOTDatasetRandomSampler:
    def __init__(self, dataset, data_getter, rng_engine, number_of_objects=1, frame_range_size=None, sampling_allow_invalid_bounding_box=False, sampling_allow_duplication=True,
                 sampling_allow_insufficiency=True, sort_result=False):
        self.dataset_sampler = SingleObjectTrackingDatasetSampler(dataset, rng_engine)
        self.sequence_sampler = SingleObjectTrackingDatasetSequenceSampler(data_getter, number_of_objects, frame_range_size, sampling_allow_invalid_bounding_box, sampling_allow_duplication, sampling_allow_insufficiency, sort_result, rng_engine)

    def get_next(self):
        return self.sequence_sampler(self.dataset_sampler.get_next_random_track())
