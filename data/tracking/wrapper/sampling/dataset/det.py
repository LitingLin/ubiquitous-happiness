from Dataset.DET.Storage.MemoryMapped.dataset import DetectionDataset_MemoryMapped
import numpy as np


class DetectionDatasetSampler:
    def __init__(self, dataset: DetectionDataset_MemoryMapped, sequence_sampler):
        self.dataset = dataset
        self.sequence_sampler = sequence_sampler

    def get_random_track(self):
        index_of_image = np.random.randint(0, len(self.dataset))
        return self.get_random_track_in_sequence(index_of_image)

    def get_random_track_in_sequence(self, index):
        image = self.dataset[index]
        assert image.has_bounding_box()
        assert len(image) > 0
        index_of_object = np.random.randint(0, len(image))
        object_ = image[index_of_object]
        return self.sequence_sampler(image, object_)

    def get_track_in_sequence(self, index_of_sequence: int, index_of_track_in_sequence: int):
        image = self.dataset[index_of_sequence]
        assert image.has_bounding_box()
        object_ = image[index_of_track_in_sequence]
        return self.sequence_sampler(image, object_)

    def get_number_of_tracks_in_sequence(self, index_of_sequence: int):
        image = self.dataset[index_of_sequence]
        return len(image)

    def __len__(self):
        return len(self.dataset)
