from Dataset._base_seed import BaseSeed
from Dataset.SOT.Base.dataset import SingleObjectTrackingDataset
from Dataset.SOT.Base.MemoryMapped.dataset import SingleObjectTrackingDataset_MemoryMapped
from .Filters.apply_filters import apply_filters


class SingleObjectTrackingDatasetFactory:
    def __init__(self, seed: BaseSeed):
        self.seed = seed
        self.dataset = None

    def construct(self, filters=None):
        dataset = SingleObjectTrackingDataset()
        constructor = dataset.getConstructor()
        constructor.setDataVersion(self.seed.data_version)
        constructor.setDataSplit(self.seed.data_split)
        constructor.setDatasetName(self.seed.name)
        constructor.setRootPath(self.seed.root_path)
        if filters is not None:
            dataset.filters = filters
        if constructor.tryLoadCache():
            return dataset

        dataset.filters = []
        if filters is None or not constructor.tryLoadCache():
            self.seed.construct(constructor)

            constructor.performStatistic()
            constructor.makeCache()

        if filters is None:
            return dataset
        else:
            return apply_filters(dataset, filters)

    def construct_memory_mapped(self, filters=None):
        dataset = SingleObjectTrackingDataset_MemoryMapped()
        constructor = dataset.getConstructor()
        constructor.setDatasetName(self.seed.name)
        constructor.setRootPath(self.seed.root_path)
        constructor.setDataSplit(self.seed.data_split)
        constructor.setDataVersion(self.seed.data_version)
        if filters is not None:
            dataset.filters = filters
        if constructor.tryLoadCache():
            return dataset

        base_dataset = self.construct(filters)

        constructor.loadFrom(base_dataset)
        constructor.makeCache()
        return dataset
