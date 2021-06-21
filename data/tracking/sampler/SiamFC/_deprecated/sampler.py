import numpy as np
from Dataset.SOT.Storage.MemoryMapped.dataset import SingleObjectTrackingDataset_MemoryMapped
from Dataset.MOT.Storage.MemoryMapped.dataset import MultipleObjectTrackingDataset_MemoryMapped
from Dataset.DET.Storage.MemoryMapped.dataset import DetectionDataset_MemoryMapped

from data.tracking.sampler._sampler.sequence.SiamFC.DET import \
    do_sampling_in_detection_dataset_image, get_one_random_sample_in_detection_dataset_image
from data.tracking.sampler._sampler.sequence.SiamFC.SOT import \
    do_sampling_in_single_object_tracking_dataset_sequence, \
    do_negative_sampling_in_single_object_tracking_dataset_sequence, \
    get_one_random_sample_in_single_object_tracking_dataset_sequence
from data.tracking.sampler._sampler.sequence.SiamFC.MOT import \
    do_sampling_in_multiple_object_tracking_dataset_sequence, \
    do_negative_sampling_in_multiple_object_tracking_dataset_sequence, \
    get_one_random_sample_in_multiple_object_tracking_dataset_sequence

from data.tracking.sampler.SiamFC.type import SiamesePairSamplingMethod


class SOTTrackingSiameseIterableDatasetSampler:
    def __init__(self, datasets, negative_sample_ratio, enforce_fine_positive_sample, sampling_method: SiamesePairSamplingMethod, datasets_sampling_parameters=None, datasets_sampling_weight=None, data_processor=None):
        self.datasets = datasets

        self.dataset_lengths = [len(dataset) for dataset in datasets]
        self.datasets_sampling_weight = datasets_sampling_weight
        self.negative_sample_ratio = negative_sample_ratio
        self.enforce_fine_positive_sample = enforce_fine_positive_sample
        raise NotImplementedError
        self.sampling_method = sampling_method
        self.data_processor = data_processor
        self.datasets_sampling_parameters = datasets_sampling_parameters

        self.current_index_of_dataset = None
        self.current_index_of_sequence = None
        self.current_is_sampling_positive_sample = None

    def move_next(self, rng_engine: np.random.Generator):
        index_of_dataset = rng_engine.choice(np.arange(len(self.datasets)), p=self.datasets_sampling_weight)
        if self.negative_sample_ratio == 0:
            is_negative = False
        else:
            is_negative = rng_engine.random() < self.negative_sample_ratio

        index_of_sequence = rng_engine.integers(0, self.dataset_lengths[index_of_dataset])

        self.current_index_of_dataset = index_of_dataset
        self.current_is_sampling_positive_sample = not is_negative
        self.current_index_of_sequence = index_of_sequence

    def _pick_random_object_as_negative_sample(self, rng_engine: np.random.Generator):
        index_of_dataset = rng_engine.choice(np.arange(len(self.datasets)), p=self.datasets_sampling_weight)
        dataset = self.datasets[index_of_dataset]
        index_of_sequence = rng_engine.integers(0, len(dataset))
        sequence = dataset[index_of_sequence]
        if isinstance(dataset, DetectionDataset_MemoryMapped):
            data = get_one_random_sample_in_detection_dataset_image(sequence, rng_engine)
        elif isinstance(dataset, SingleObjectTrackingDataset_MemoryMapped):
            data = get_one_random_sample_in_single_object_tracking_dataset_sequence(sequence, rng_engine)
        elif isinstance(dataset, MultipleObjectTrackingDataset_MemoryMapped):
            data = get_one_random_sample_in_multiple_object_tracking_dataset_sequence(sequence, rng_engine)
        else:
            raise NotImplementedError
        return data

    def do_sampling(self, rng_engine: np.random.Generator):
        dataset = self.datasets[self.current_index_of_dataset]
        sequence = dataset[self.current_index_of_sequence]

        frame_range = 100
        if self.datasets_sampling_parameters is not None:
            sampling_parameter = self.datasets_sampling_parameters[self.current_index_of_dataset]
            if 'frame_range' in sampling_parameter:
                frame_range = sampling_parameter['frame_range']
        if isinstance(dataset, (SingleObjectTrackingDataset_MemoryMapped, MultipleObjectTrackingDataset_MemoryMapped)):
            if sequence.has_fps():
                fps = sequence.get_fps()
                frame_range = int(round(fps / 30 * frame_range))

        if self.current_is_sampling_positive_sample:
            if isinstance(dataset, DetectionDataset_MemoryMapped):
                z_image, z_bbox = do_sampling_in_detection_dataset_image(sequence, rng_engine)
                data = (z_image, z_bbox, z_image, z_bbox, True)
            elif isinstance(dataset, (SingleObjectTrackingDataset_MemoryMapped, MultipleObjectTrackingDataset_MemoryMapped)):
                if isinstance(dataset, SingleObjectTrackingDataset_MemoryMapped):
                    sampled_data, is_positive = do_sampling_in_single_object_tracking_dataset_sequence(sequence, frame_range, rng_engine)
                else:
                    sampled_data, is_positive = do_sampling_in_multiple_object_tracking_dataset_sequence(sequence, frame_range, rng_engine)
                if is_positive == 0:
                    data = (sampled_data[0][0], sampled_data[0][1], sampled_data[0][0], sampled_data[0][1], True)
                else:
                    data = (sampled_data[0][0], sampled_data[0][1], sampled_data[1][0], sampled_data[1][1], is_positive == 1)
            else:
                raise NotImplementedError
        else:
            if isinstance(dataset, DetectionDataset_MemoryMapped):
                z_image, z_bbox = do_sampling_in_detection_dataset_image(sequence, rng_engine)
                x_image, x_bbox = self._pick_random_object_as_negative_sample(rng_engine)
                data = (z_image, z_bbox, x_image, x_bbox, False)
            elif isinstance(dataset, (SingleObjectTrackingDataset_MemoryMapped, MultipleObjectTrackingDataset_MemoryMapped)):
                if isinstance(dataset, SingleObjectTrackingDataset_MemoryMapped):
                    sampled_data = do_negative_sampling_in_single_object_tracking_dataset_sequence(sequence, frame_range, rng_engine)
                else:
                    sampled_data = do_negative_sampling_in_multiple_object_tracking_dataset_sequence(sequence, frame_range, rng_engine)
                if len(sampled_data) == 1:
                    x_image, x_bbox = self._pick_random_object_as_negative_sample(rng_engine)
                    data = (sampled_data[0][0], sampled_data[0][1], x_image, x_bbox, False)
                else:
                    data = (sampled_data[0][0], sampled_data[0][1], sampled_data[1][0], sampled_data[1][1], False)
            else:
                raise NotImplementedError
        if self.data_processor is not None:
            data = self.data_processor(*data)
        return data
