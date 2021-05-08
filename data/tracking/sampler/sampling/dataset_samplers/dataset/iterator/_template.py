class DatasetSamplerTemplate:
    def __init__(self, indexer, getter):
        self.indexer = indexer
        self.getter = getter

    def current(self):
        return self.getter[self.indexer.current()]

    def move_next(self):
        return self.indexer.move_next()

    def reset(self):
        self.indexer.reset()

    def current_position(self):
        return self.indexer.current()

    def length(self):
        return self.indexer.length()
