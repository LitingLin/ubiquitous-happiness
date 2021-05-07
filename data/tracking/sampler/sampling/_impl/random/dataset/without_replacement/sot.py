from .indexing import RandomSamplingWithoutReplacement


class SingleObjectTrackingDatasetSampler:
    def __init__(self, dataset, rng_engine):
        self.dataset = dataset
        self.rng_engine = rng_engine
        self.random_indexer = RandomSamplingWithoutReplacement(len(dataset), rng_engine)

    def get_next_random_track(self):
        index = self.random_indexer.next()
        sequence = self.dataset[index]
        return sequence
