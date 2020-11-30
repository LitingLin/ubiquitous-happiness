from .Base.dataset import MultipleObjectTrackingDataset
from Dataset.SOT.Base.Numpy.dataset import SingleObjectTrackingDataset_Numpy
from Dataset.SOT.Base.MemoryMapped.dataset import SingleObjectTrackingDataset_MemoryMapped
from .Converter.to_sot_numpy import to_sot_numpy
from .Converter.to_sot_memory_mapped import to_sot_memory_mapped
from .Filters.apply_filters import apply_filters


class MultipleObjectTrackingDatasetFactory:
    def __init__(self, seed):
        self.seed = seed

    def construct(self, filters=None):
        dataset = MultipleObjectTrackingDataset()
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
    """
    def _constructDetectionDataset(self):
        from Dataset.Detection.Base.dataset import DetectionDataset
        dataset = DetectionDataset()
        constructor = dataset.getConstructor()
        constructor.setDataSplit(self.seed.data_split)
        constructor.setDatasetName(self.seed.name)
        constructor.setDataVersion(self.seed.data_version)
        constructor.setRootPath(self.seed.root_path)
        if constructor.tryLoadCache():
            return dataset
        mot_dataset = self.construct()
        for sequence in mot_dataset:
            for index_of_frame, frame in enumerate(sequence):
                constructor.beginInitializeImage()
                constructor.setImageName(sequence.getName() + '_' + str(index_of_frame))
                constructor.setImagePath(frame.getImagePath())
                for object_ in frame:
                    is_present = object_.getAttributeIsPresent()
                    category_name = object_.getCategoryName()
                    constructor.addObject(object_.getBoundingBox(), category_name, is_present)
                constructor.endInitializeImage()
        constructor.performStatistic()
        constructor.makeCache()
        return dataset

    def constructDetectionDataset(self, filters=None):
        dataset = self._constructDetectionDataset()
        if filters is None:
            return dataset
        from Dataset.Detection.Filters.apply_filters import apply_filters
        return apply_filters(dataset, filters)

    def constructDetectionMemoryMappedVersion(self, filters=None):
        from Dataset.Detection.Base.MemoryMapped.dataset import DetectionDataset_MemoryMapped
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

        base_dataset = self.constructDetectionDataset(filters)
        constructor.loadFrom(base_dataset)
        constructor.makeCache()
        return dataset

    def constructSingleObjectTrackingCompactVersion(self):
        if self.sot_compact_dataset is not None:
            return self.sot_compact_dataset

        dataset = SingleObjectTrackingDataset_Numpy()
        constructor = dataset.getConstructor()
        constructor.setName(self.seed.name + '-flatten')
        constructor.setDataSplit(self.seed.data_split)
        constructor.setRootPath(self.seed.root_path)
        constructor.setDataVersion(self.seed.data_version)
        if constructor.tryLoadCache():
            self.sot_compact_dataset = dataset
            return dataset

        self.construct()

        to_sot_numpy(self.dataset, dataset)
        constructor.makeCache()
        self.sot_compact_dataset = dataset
        return dataset

    def constructSingleObjectTrackingMemoryMappedVersion(self):
        if self.sot_memory_mapped_dataset is not None:
            return self.sot_memory_mapped_dataset

        dataset = SingleObjectTrackingDataset_MemoryMapped()
        constructor = dataset.getConstructor()
        constructor.setDatasetName(self.seed.name + '-flatten-memory_mapped')
        constructor.setDataSplit(self.seed.data_split)
        constructor.setRootPath(self.seed.root_path)
        constructor.setDataVersion(self.seed.data_version)
        if constructor.tryLoadCache():
            self.sot_memory_mapped_dataset = dataset
            return dataset

        self.construct()

        to_sot_memory_mapped(self.dataset, dataset)

        constructor.makeCache()
        self.sot_memory_mapped_dataset = dataset
        return dataset
    """
