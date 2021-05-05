from .indexing import RandomSamplingWithoutReplacement


class DetectionDatasetSampler:
    def __init__(self, dataset, rng_engine):
        self.dataset = dataset
        self.random_indexer = RandomSamplingWithoutReplacement(len(dataset), rng_engine)

    def get_next_random_track(self):
        index = self.random_indexer.next()
        return self.dataset[index]
