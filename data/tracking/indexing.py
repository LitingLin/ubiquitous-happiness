from Dataset.DET.Storage.MemoryMapped.dataset import DetectionDataset_MemoryMapped
from Dataset.MOT.Storage.MemoryMapped.dataset import MultipleObjectTrackingDataset_MemoryMapped
from Dataset.SOT.Storage.MemoryMapped.dataset import SingleObjectTrackingDataset_MemoryMapped
from enum import Enum, auto
import numpy as np


class DatasetWeightingStrategy(Enum):
    absolute = auto()
    relative = auto()


class _ConcatenatedDatasetPositioningHelper:
    def __init__(self):
        self.dataset_sizes = []

    def register(self, size):
        self.dataset_sizes.append(size)

    def __call__(self, index):
        if index < 0:
            raise IndexError
        for index_of_dataset, size in enumerate(self.dataset_sizes):
            if index < size:
                return index_of_dataset, index
            index -= size
        raise IndexError


class TrackingDatasetIndexing:
    def __init__(self, datasets, weighting_strategy=DatasetWeightingStrategy.relative, repeat_times=None, total_length=None, image_dataset_weight=None, video_dataset_weight=None, rng_engine=np.random):
        dataset_sizes = []
        dataset_weights = []

        if image_dataset_weight is None:
            image_dataset_weight = 1
        if video_dataset_weight is None:
            video_dataset_weight = 1
        r = min(image_dataset_weight, video_dataset_weight)
        image_dataset_weight /= r
        video_dataset_weight /= r

        for dataset in datasets:
            assert isinstance(dataset, (DetectionDataset_MemoryMapped, SingleObjectTrackingDataset_MemoryMapped, MultipleObjectTrackingDataset_MemoryMapped)), f'unsupported type {type(dataset)}'

            sampling_weight = 1
            if dataset.has_attribute('sampling_weight'):
                sampling_weight = dataset.get_attribute('sampling_weight')
            if isinstance(dataset, DetectionDataset_MemoryMapped):
                sampling_weight *= image_dataset_weight
            else:
                sampling_weight *= video_dataset_weight
            if dataset.has_attribute('length'):
                length = dataset.get_attribute('length')
            else:
                length = len(dataset)
            dataset_weights.append(sampling_weight)
            dataset_sizes.append(length)

        if repeat_times is not None:
            dataset_sizes = [size * repeat_times for size in dataset_sizes]

        if weighting_strategy == DatasetWeightingStrategy.relative:
            s = sum(dataset_sizes)
            dataset_length_ratios = [size / s for size in dataset_sizes]
            dataset_weights = [dataset_weights[i] * dataset_length_ratios[i] for i in range(len(dataset_weights))]
        if total_length is None:
            r = min(dataset_weights)
            dataset_weights = [weight / r for weight in dataset_weights]
            dataset_sizes = [round(dataset_sizes[i] * dataset_weights[i]) for i in range(len(dataset_sizes))]
            total_length = sum(dataset_sizes)
        else:
            m = sum(dataset_weights)
            dataset_weights = [weight / m for weight in dataset_weights]
            dataset_sizes = [round(total_length * weight) for weight in dataset_weights]
            total_length = sum(dataset_sizes)

        # eliminate zero size
        dataset_index_relative_real_map = {}
        valid_dataset_count = 0
        zero_size_dataset_indices = []
        for index, dataset_size in enumerate(dataset_sizes):
            if dataset_size > 0:
                dataset_index_relative_real_map[valid_dataset_count] = index
                valid_dataset_count += 1
            else:
                zero_size_dataset_indices.append(index)
        self.dataset_index_relative_real_map = dataset_index_relative_real_map
        datasets = [i for j, i in enumerate(datasets) if j not in zero_size_dataset_indices]
        dataset_sizes = [i for j, i in enumerate(dataset_sizes) if j not in zero_size_dataset_indices]

        self.datasets = datasets
        self.dataset_sizes = dataset_sizes
        self.positioning_helper = _ConcatenatedDatasetPositioningHelper()
        self.total_length = total_length
        for dataset in datasets:
            self.positioning_helper.register(len(dataset))
        self.rng_engine = rng_engine

    def __len__(self):
        return self.total_length

    def __getitem__(self, index: int):
        index = self.indices[index]
        index_of_dataset, index_in_dataset = self.positioning_helper(index)
        return self.dataset_index_relative_real_map[index_of_dataset], index_in_dataset

    def shuffle(self):
        offset = 0
        indices = []
        for dataset, size in zip(self.datasets, self.dataset_sizes):
            index = np.arange(offset, offset + len(dataset))
            self.rng_engine.shuffle(index)
            while index.shape[0] < size:
                indices.append(index)
                size -= index.shape[0]
            if size != 0:
                indices.append(index[: size])
            offset += len(dataset)
        self.indices = np.concatenate(indices)
        self.rng_engine.shuffle(self.indices)

    def sequential(self):
        offset = 0
        indices = []
        for dataset, size in zip(self.datasets, self.dataset_sizes):
            index = np.arange(offset, offset + len(dataset))
            while index.shape[0] < size:
                indices.append(index)
                size -= index.shape[0]
            if size != 0:
                indices.append(index[: size])
            offset += len(dataset)
        self.indices = np.concatenate(indices)
