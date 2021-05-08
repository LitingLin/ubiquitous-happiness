from .indexing import RandomSamplingWithoutReplacement


class _DatasetSampler:
    def __init__(self, dataset, rng_engine):
        self.dataset = dataset
        self.random_indexer = RandomSamplingWithoutReplacement(len(dataset), rng_engine)

    def move_next(self):
        return self.random_indexer.move_next()

    def reset(self):
        self.random_indexer.reset()

    def current_position(self):
        return self.random_indexer.current()

    def length(self):
        return self.random_indexer.length()
