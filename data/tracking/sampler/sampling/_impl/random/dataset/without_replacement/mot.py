from .indexing import RandomSamplingWithoutReplacement


class MultipleObjectTrackingDatasetSampler:
    def __init__(self, dataset, rng_engine):
        self.dataset = dataset
        self.rng_engine = rng_engine
        self.random_indexer = RandomSamplingWithoutReplacement(len(dataset), rng_engine)

    def get_next_random_track(self):
        index = self.random_indexer.next()
        sequence = self.dataset[index]

        number_of_objects = sequence.get_number_of_objects()
        index_of_object_in_sequence = self.rng_engine.randint(0, number_of_objects)
        sequence_object = sequence.get_object(index_of_object_in_sequence)
        return sequence, sequence_object
