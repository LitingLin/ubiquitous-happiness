from ._common import _DatasetSampler


class MultipleObjectTrackingDatasetSampler(_DatasetSampler):
    def __init__(self, dataset, rng_engine):
        super(MultipleObjectTrackingDatasetSampler, self).__init__(dataset, rng_engine)
        self.rng_engine = rng_engine

    def current(self):
        sequence = self.dataset[self.random_indexer.current()]
        number_of_objects = sequence.get_number_of_objects()
        index_of_object_in_sequence = self.rng_engine.randint(0, number_of_objects)
        sequence_object = sequence.get_object(index_of_object_in_sequence)
        return sequence, sequence_object
