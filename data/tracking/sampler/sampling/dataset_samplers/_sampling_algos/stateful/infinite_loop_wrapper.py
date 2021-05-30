class Sampling_InfinteLoopWrapper:
    def __init__(self, sampling_algo):
        self.sampling_algo = sampling_algo
        assert self.sampling_algo.move_next()

    def get(self):
        return self.sampling_algo.current()

    def move_next(self):
        if not self.sampling_algo.move_next():
            self.sampling_algo.reset()
            assert self.sampling_algo.move_next()
