class InfiniteLooper:
    def __init__(self, iterator):
        self.iterator = iterator

    def get_next(self):
        current = self.iterator.current()
        if not self.iterator.move_next():
            self.iterator.reset()
        return current
