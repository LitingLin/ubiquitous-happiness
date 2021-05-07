import numpy as np


class RandomSamplingWithoutReplacement:
    def __init__(self, length, rng_engine):
        self.rng_engine = rng_engine
        self.indices = np.arange(0, length)

    def _shuffle(self):
        self.position = 0
        self.rng_engine.shffule(self.indices)

    def next(self):
        if self.position >= len(self.indices):
            self._shuffle()
        position = self.position
        self.position += 1
        return position
