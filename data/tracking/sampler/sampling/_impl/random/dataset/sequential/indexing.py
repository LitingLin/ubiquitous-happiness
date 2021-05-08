class SequentialIndexing:
    def __init__(self, length):
        assert length > 0
        self.position = 0
        self.length_ = length

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
