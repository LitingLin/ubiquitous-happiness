class DatasetsRandomSampler:
    def __init__(self, sampling_weights, rng_engine):
        self.rng_engine = rng_engine
        self.sampling_weights = sampling_weights

    def get_next_dataset_index(self):
        return self.rng_engine.choice(len(self.sampling_weights), p=self.sampling_weights)
