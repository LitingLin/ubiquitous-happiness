import numpy as np


class Sampling_RandomSamplingWithoutReplacement:
    def __init__(self, length, seed):
        self.position = 0
        self.rng_seed = seed
        self._shuffle(length)

    def _shuffle(self, length):
        rng_engine = np.random.default_rng(self.rng_seed)
        self.indices = np.arange(length)
        rng_engine.shuffle(self.indices)

    @staticmethod
    def restore_from_state(state):
        position, length, seed = state
        sampler = Sampling_RandomSamplingWithoutReplacement(length, seed)
        sampler.position = position
        return sampler

    def get_state(self):
        return self.position, len(self.indices), self.rng_seed

    def move_next(self):
        if self.position + 1 >= len(self.indices):
            return False
        self.position += 1
        return True

    def current(self):
        return self.position

    def reset(self):
        self.rng_seed += 1
        self.position = 0
        self._shuffle(len(self.indices))

    def length(self):
        return len(self.indices)
