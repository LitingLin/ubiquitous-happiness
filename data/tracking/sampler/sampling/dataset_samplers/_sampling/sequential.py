class Sampling_SequentialIndexing:
    def __init__(self, length):
        assert length > 0
        self.position = 0
        self.length_ = length

    @staticmethod
    def restore_from_state(state):
        position, length = state
        sampler = Sampling_SequentialIndexing(length)
        sampler.position = position
        return sampler

    def get_state(self):
        return self.position, self.length_

    def move_next(self):
        if self.position + 1 >= self.length_:
            return False
        self.position += 1
        return True

    def current(self):
        return self.position

    def reset(self):
        self.position = 0

    def length(self):
        return self.length_
