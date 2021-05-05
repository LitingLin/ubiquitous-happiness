from enum import Enum, auto
import numpy as np
from Dataset.DET.Storage.MemoryMapped.dataset import DetectionDataset_MemoryMapped
from Dataset.MOT.Storage.MemoryMapped.dataset import MultipleObjectTrackingDataset_MemoryMapped
from Dataset.SOT.Storage.MemoryMapped.dataset import SingleObjectTrackingDataset_MemoryMapped


class DatasetWeightingStrategy(Enum):
    absolute = auto()
    relative = auto()


class TrackingDatasetSiameseRandomSampler:
    def __init__(self, datasets, weighting_strategy=DatasetWeightingStrategy.relative, length=None, image_dataset_weight=None, video_dataset_weight=None, rng_engine=np.random):
        for dataset in datasets:
            assert dataset in (DetectionDataset_MemoryMapped, SingleObjectTrackingDataset_MemoryMapped,
                               MultipleObjectTrackingDataset_MemoryMapped)

        self.datasets = datasets
        self.dataset_sampler = {}
        self.track_sampler = {}
        for dataset in datasets:



        pass


    def __iter__(self):
        pass


    def __next__(self):
        pass


    def __len__(self):
        pass
