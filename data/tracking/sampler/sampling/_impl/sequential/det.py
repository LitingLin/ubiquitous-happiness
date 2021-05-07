from data.tracking.sampler.sampling._impl.random.track.random.det import



class DETDatasetSequentialSampler:
    def __init__(self, dataset):
        self.dataset = dataset

    def get_next(self):
        pass

    def __len__(self):
