from .._base_seed import BaseSeed
from .Base.dataset import DetectionDataset
from .Base.MemoryMapped.dataset import DetectionDataset_MemoryMapped
from .Filters.apply_filters import apply_filters

class DetectionDatasetFactory:
    def __init__(self, seed: BaseSeed):
        self.seed = seed

    def construct(self, filters=None):
        dataset = DetectionDataset()
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

    def constructMemoryMappedVersion(self, filters=None):
        dataset = DetectionDataset_MemoryMapped()
        constructor = dataset.getConstructor()
        constructor.setDatasetName(self.seed.name)
        constructor.setDataVersion(self.seed.data_version)
        constructor.setDataSplit(self.seed.data_split)
        constructor.setRootPath(self.seed.root_path)
        if filters is not None:
            dataset.filters = filters
        if constructor.tryLoadCache():
            return dataset

        base_dataset = self.construct(filters)
        constructor.loadFrom(base_dataset)
        constructor.makeCache()
        return dataset

    def constructSingleObjectDetection_MemoryMappedVersion(self, filters=None):
        return self.constructMemoryMappedVersion(filters).getFlattenView()
