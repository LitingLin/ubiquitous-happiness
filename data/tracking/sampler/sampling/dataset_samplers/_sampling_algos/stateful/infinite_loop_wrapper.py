class Sampling_InfinteLoopWrapper:
    def __init__(self, sampling_algo):
        self.sampling_algo = sampling_algo

    def restore_from_state(self, state):
        self.sampling_algo.restore_from_state(state)

    def get_state(self):
        return self.sampling_algo.get_state()

    def current(self):
        return self.sampling_algo.current()

    def move_next(self):
        if not self.sampling_algo.move_next():
            self.sampling_algo.reset()
