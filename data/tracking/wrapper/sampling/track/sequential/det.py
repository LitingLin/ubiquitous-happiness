from Dataset.DET.Storage.MemoryMapped.dataset import DetectionDatasetImage_MemoryMapped, DetectionDatasetObject_MemoryMapped


class _DETTrackIterator:
    def __init__(self, data_getter, image: DetectionDatasetImage_MemoryMapped, object_: DetectionDatasetObject_MemoryMapped):
        self.data_getter = data_getter
        self.image = image
        self.object_ = object_
        self.consumed = False

    def __next__(self):
        if not self.consumed:
            self.consumed = True
            return self.data_getter(self.image, self.object_)
        else:
            raise StopIteration


class DetectionDatasetTrackIteratorGenerator:
    def __init__(self, data_getter, image: DetectionDatasetImage_MemoryMapped, object_: DetectionDatasetObject_MemoryMapped):
        self.data_getter = data_getter
        self.image = image
        self.object_ = object_

    def __iter__(self):
        return _DETTrackIterator(self.data_getter, self.image, self.object_)

    def __len__(self):
        return 1
