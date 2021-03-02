from Dataset.DET.Storage.MemoryMapped.dataset import DetectionDataset_MemoryMapped, DetectionDatasetImage_MemoryMapped, DetectionDatasetObject_MemoryMapped
import numpy as np


class _DETTrackIterator:
    def __init__(self, image: DetectionDatasetImage_MemoryMapped, object_: DetectionDatasetObject_MemoryMapped):
        self.image = image
        self.object_ = object_
        self.consumed = False

    def __next__(self):
        if not self.consumed:
            self.consumed = True
            return self.image.get_image_path(), self.object_.get_bounding_box(), self.object_.get_bounding_box_validity_flag() if self.object_.has_bounding_box_validity_flag() else True
        else:
            raise StopIteration


class DetectionDatasetTrackIteratorGenerator:
    def __init__(self, image: DetectionDatasetImage_MemoryMapped, object_: DetectionDatasetObject_MemoryMapped):
        self.image = image
        self.object_ = object_

    def __iter__(self):
        return _DETTrackIterator(self.image, self.object_)
