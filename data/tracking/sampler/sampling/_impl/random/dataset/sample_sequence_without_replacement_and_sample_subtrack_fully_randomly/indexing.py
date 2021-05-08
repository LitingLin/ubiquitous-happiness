import numpy as np


class RandomSamplingWithoutReplacement:
    def __init__(self, length, rng_engine):
        assert length > 0
        self.rng_engine = rng_engine
        self.indices = np.arange(0, length)
        self.reset()

    def _shuffle(self):
        self.rng_engine.shffule(self.indices)

    def move_next(self):
        if self.position + 1 >= len(self.indices):
            return False
        self.position += 1
        return True

    def current(self):
        return self.position

    def reset(self):
        self.position = 0
        self._shuffle()

    def length(self):
        return len(self.indices)
