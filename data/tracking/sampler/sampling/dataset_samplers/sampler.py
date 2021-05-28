



class DatasetSampler:
    def __init__(self, dataset, dataset_sampler, sequence_sampler, track_sampler):
        self.dataset = dataset
        self.dataset_sampler = dataset_sampler
        self.sequence_sampler = sequence_sampler
        self.track_sampler = track_sampler

    def get_state(self):
        return self.dataset_sampler.get_state(), self.dataset_sampler.get_state(), self.dataset_sampler.get_state()

    def restore_from_state(self):
        pass

    def move_next(self):
        pass

    def current(self):
        sequence_index = self.dataset_sampler.current()
        track_index = self.sequence_sampler.current()
        object_ = self.track_sampler.current()
        return object_
