from enum import Enum, auto
import numpy as np
from Dataset.DET.Storage.MemoryMapped.dataset import DetectionDataset_MemoryMapped
from Dataset.MOT.Storage.MemoryMapped.dataset import MultipleObjectTrackingDataset_MemoryMapped
from Dataset.SOT.Storage.MemoryMapped.dataset import SingleObjectTrackingDataset_MemoryMapped

from data.tracking.sampler.getter.standard.sot import single_object_tracking_dataset_standard_data_getter
from data.tracking.sampler.getter.standard.det import detection_dataset_standard_data_getter
from data.tracking.sampler.getter.standard.mot import multiple_object_tracking_dataset_standard_data_getter
from data.tracking.sampler.sampling.random.dataset.sot import SOTDatasetRandomSampler
from data.tracking.sampler.sampling.random.dataset.det import DETDatasetRandomSampler
from data.tracking.sampler.sampling.random.dataset.mot import MOTDatasetRandomSampler
from data.tracking.sampler.sampling._impl.random.datasets.sampler import DatasetsRandomSampler
from decimal import Decimal


class DatasetWeightingStrategy(Enum):
    absolute = auto()
    relative = auto()


class TrackingDatasetSiameseRandomSampler:
    def __init__(self, datasets, sample_post_processor,
                 dataset_sampling_weights, dataset_sampling_frame_ranges,
                 weighting_strategy=DatasetWeightingStrategy.relative,
                 image_dataset_weight=None, video_dataset_weight=None,
                 number_of_sampling_objects=1,
                 positive_sampling_allow_invalid_bounding_box=True, positive_sampling_allow_duplication=False, positive_sampling_allow_insufficiency=True, positive_sampling_sort_result=False,
                 negative_sampling_ratio=0,
                 rng_engine=np.random):
        dataset_sampling_weights = [Decimal(weight) for weight in dataset_sampling_weights]
        total_length = 0
        for index_of_dataset, dataset in enumerate(datasets):
            assert isinstance(dataset, (DetectionDataset_MemoryMapped, SingleObjectTrackingDataset_MemoryMapped,
                               MultipleObjectTrackingDataset_MemoryMapped))

            if weighting_strategy == DatasetWeightingStrategy.relative:
                dataset_sampling_weights[index_of_dataset] = dataset_sampling_weights[index_of_dataset] * len(dataset)
            if isinstance(dataset, DetectionDataset_MemoryMapped):
                if image_dataset_weight is not None:
                    dataset_sampling_weights[index_of_dataset] = dataset_sampling_weights[index_of_dataset] * image_dataset_weight
            elif isinstance(dataset, (SingleObjectTrackingDataset_MemoryMapped, MultipleObjectTrackingDataset_MemoryMapped)):
                if video_dataset_weight is not None:
                    dataset_sampling_weights[index_of_dataset] = dataset_sampling_weights[index_of_dataset] * video_dataset_weight

            total_length += len(dataset)

        weight_sum = sum(dataset_sampling_weights)
        dataset_sampling_weights = [weight / weight_sum for weight in dataset_sampling_weights]
        dataset_sampling_weights = [float(weight) for weight in dataset_sampling_weights]
        dataset_sampling_weights = np.array(dataset_sampling_weights)

        self.datasets_sampler = DatasetsRandomSampler(dataset_sampling_weights, rng_engine)

        positive_samplers = []
        for index_of_dataset, dataset in enumerate(datasets):
            if isinstance(dataset, DetectionDataset_MemoryMapped):
                positive_samplers.append(DETDatasetRandomSampler(dataset, detection_dataset_standard_data_getter, rng_engine, positive_sampling_allow_invalid_bounding_box))
            elif isinstance(dataset, SingleObjectTrackingDataset_MemoryMapped):
                positive_samplers.append(
                    SOTDatasetRandomSampler(dataset, single_object_tracking_dataset_standard_data_getter, rng_engine,
                                            number_of_sampling_objects, dataset_sampling_frame_ranges[index_of_dataset],
                                            positive_sampling_allow_invalid_bounding_box, positive_sampling_allow_duplication, positive_sampling_allow_insufficiency, positive_sampling_sort_result))
            elif isinstance(dataset, MultipleObjectTrackingDataset_MemoryMapped):
                positive_samplers.append(
                    MOTDatasetRandomSampler(dataset, multiple_object_tracking_dataset_standard_data_getter, rng_engine,
                                            number_of_sampling_objects, dataset_sampling_frame_ranges[index_of_dataset],
                                            positive_sampling_allow_invalid_bounding_box, positive_sampling_allow_duplication, positive_sampling_allow_insufficiency, positive_sampling_sort_result))
        self.positive_samplers = positive_samplers

        assert 0 <= negative_sampling_ratio <= 1

        self.negative_sampling_ratio = negative_sampling_ratio
        if negative_sampling_ratio > 0:
            negative_samplers = []
            for dataset in datasets:
                if isinstance(dataset, DetectionDataset_MemoryMapped):
                    positive_samplers.append(
                        DETDatasetRandomSampler(dataset, detection_dataset_standard_data_getter, rng_engine, True))
                elif isinstance(dataset, SingleObjectTrackingDataset_MemoryMapped):
                    positive_samplers.append(
                        SOTDatasetRandomSampler(dataset, single_object_tracking_dataset_standard_data_getter,
                                                rng_engine, 1, 1, True))
                elif isinstance(dataset, MultipleObjectTrackingDataset_MemoryMapped):
                    positive_samplers.append(
                        MOTDatasetRandomSampler(dataset, multiple_object_tracking_dataset_standard_data_getter,
                                                rng_engine, 1, 1, True))
            self.negative_samplers = negative_samplers
        self.number_of_sampling_objects = number_of_sampling_objects
        self.length = total_length
        self.rng_engine = rng_engine
        self.sample_post_processor = sample_post_processor

    def _get_next_positive_sample(self):
        index_of_dataset = self.datasets_sampler.get_next_dataset_index()
        return self.positive_samplers[index_of_dataset].get_next()

    def _get_next_negative_sample(self):
        data = []
        for index_of_object in range(self.number_of_sampling_objects):
            dataset_index = self.datasets_sampler.get_next_dataset_index()
            data.extend(self.negative_samplers[dataset_index].get_next())
        return data

    def get_next(self):
        if self.negative_sampling_ratio == 0:
            return self.sample_post_processor(self._get_next_positive_sample(), True)
        else:
            is_positive = self.rng_engine.rand() > self.negative_sampling_ratio
            if is_positive:
                return self.sample_post_processor(self._get_next_positive_sample(), True)
            else:
                return self.sample_post_processor(self._get_next_negative_sample(), False)

    def get_total_length(self):
        return self.length

    def __len__(self):
        return self.length
