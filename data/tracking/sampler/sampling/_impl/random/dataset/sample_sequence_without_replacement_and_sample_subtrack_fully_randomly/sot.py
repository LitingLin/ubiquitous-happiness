from ._common import _DatasetSampler


class SingleObjectTrackingDatasetSampler(_DatasetSampler):
    def current(self):
        return self.dataset[self.random_indexer.current()]
