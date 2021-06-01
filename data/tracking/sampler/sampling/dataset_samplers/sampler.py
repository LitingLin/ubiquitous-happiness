import numpy as np


class SOTTrackingSiameseIterableDatasetSampler:
    def __init__(self, datasets, negative_sample_ratio, data_processor=None):
        self.datasets = datasets
        self.negative_sample_ratio = negative_sample_ratio
        self.data_processor = data_processor
        self.position = 0

    def get_position(self):
        return self.position

    def move_next(self, rng_engine: np.random.Generator):
        pass

    def forward_to(self, index: int, rng_engine: np.random.Generator):

        pass

    def do_sampling(self, rng_engine: np.random.Generator):
        pass

    def get_state(self):
        pass

    def load_state(self):
        pass
