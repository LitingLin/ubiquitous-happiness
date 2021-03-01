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


class _DETTrackIteratorGenerator:
    def __init__(self, image: DetectionDatasetImage_MemoryMapped, object_: DetectionDatasetObject_MemoryMapped):
        self.image = image
        self.object_ = object_

    def __iter__(self):
        return _DETTrackIterator(self.image, self.object_)


class _DETWrapper:
    def __init__(self, dataset: DetectionDataset_MemoryMapped):
        self.dataset = dataset

    def get_random_track(self):
        index_of_image = np.random.randint(0, len(self.dataset))
        return self.get_random_track_in_sequence(index_of_image)

    def get_random_track_in_sequence(self, index):
        image = self.dataset[index]
        assert image.has_bounding_box()
        assert len(image) > 0
        index_of_object = np.random.randint(0, len(image))
        object_ = image[index_of_object]
        return _DETTrackIteratorGenerator(image, object_)

    def get_track_in_sequence(self, index_of_sequence: int, index_of_track_in_sequence: int):
        image = self.dataset[index_of_sequence]
        assert image.has_bounding_box()
        object_ = image[index_of_track_in_sequence]
        return _DETTrackIteratorGenerator(image, object_)

    def get_number_of_tracks_in_sequence(self, index_of_sequence: int):
        image = self.dataset[index_of_sequence]
        return len(image)

    def __len__(self):
        return len(self.dataset)
