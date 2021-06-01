import numpy as np
from Dataset.SOT.Storage.MemoryMapped.dataset import SingleObjectTrackingDataset_MemoryMapped
from Dataset.MOT.Storage.MemoryMapped.dataset import MultipleObjectTrackingDataset_MemoryMapped
from Dataset.DET.Storage.MemoryMapped.dataset import DetectionDataset_MemoryMapped

from data.tracking.sampler.sampling.dataset_samplers._sampler.sequence.SiamFC.DET import \
    do_sampling_in_detection_dataset_image, get_one_random_sample_in_detection_dataset_image
from data.tracking.sampler.sampling.dataset_samplers._sampler.sequence.SiamFC.SOT import \
    do_sampling_in_single_object_tracking_dataset_sequence, \
    do_positive_sampling_in_single_object_tracking_dataset_sequence, \
    do_negative_sampling_in_single_object_tracking_dataset_sequence, \
    get_one_random_sample_in_single_object_tracking_dataset_sequence
from data.tracking.sampler.sampling.dataset_samplers._sampler.sequence.SiamFC.MOT import \
    do_sampling_in_multiple_object_tracking_dataset_sequence, \
    do_positive_sampling_in_multiple_object_tracking_dataset_sequence, \
    do_negative_sampling_in_multiple_object_tracking_dataset_sequence, \
    get_one_random_sample_in_multiple_object_tracking_dataset_sequence


def _build_datasets_sampler(datasets, rng_engine: np.random.Generator):
    from data.tracking.sampler.sampling.dataset_samplers._sampling_algos.stateful.random_without_replacement import SamplingAlgo_RandomSamplingWithoutReplacement
    from data.tracking.sampler.sampling.dataset_samplers._sampling_algos.stateful.infinite_loop_wrapper import Sampling_InfinteLoopWrapper

    dataset_samplers = []
    for dataset in datasets:
        dataset_samplers.append(Sampling_InfinteLoopWrapper(SamplingAlgo_RandomSamplingWithoutReplacement(len(dataset), rng_engine.integers(-1000, 1000))))
    return dataset_samplers


class SOTTrackingSiameseIterableDatasetSampler:
    def __init__(self, datasets, negative_sample_ratio, datasets_sampling_parameters=None, datasets_sampling_weight=None, data_processor=None, seed: int=0):
        self.datasets = datasets
        self.datasets_sampler = _build_datasets_sampler(datasets, np.random.Generator(np.random.PCG64(seed)))

        if datasets_sampling_weight is not None:
            datasets_sampling_weight = datasets_sampling_weight / np.sum(datasets_sampling_weight)
        self.datasets_sampling_weight = datasets_sampling_weight
        self.negative_sample_ratio = negative_sample_ratio
        self.data_processor = data_processor
        self.datasets_sampling_parameters = datasets_sampling_parameters

        self.position = -1

        self.current_index_of_dataset = None
        self.current_index_of_sequence = None
        self.current_is_sampling_positive_sample = None

    def get_position(self):
        return self.position

    def move_next(self, rng_engine: np.random.Generator):
        index_of_dataset = rng_engine.choice(np.arange(0, len(self.datasets)), p=self.datasets_sampling_weight)
        if self.negative_sample_ratio == 0:
            is_negative = False
        else:
            is_negative = rng_engine.random() < self.negative_sample_ratio

        dataset_sampler = self.datasets_sampler[index_of_dataset]
        dataset_sampler.move_next()
        index_of_sequence = dataset_sampler.get()

        self.current_index_of_dataset = index_of_dataset
        self.current_is_sampling_positive_sample = not is_negative
        self.current_index_of_sequence = index_of_sequence

        self.position += 1

    def forward_to(self, position: int, rng_engine: np.random.Generator):
        assert position >= self.position
        while self.position < position:
            self.move_next(rng_engine)

    def _pick_random_object_as_negative_sample(self, rng_engine: np.random.Generator):
        index_of_dataset = rng_engine.integers(0, len(self.datasets))
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
        return data

    def get_state(self):
        dataset_samplers_state = []
        for sampler in self.datasets_sampler:
            dataset_samplers_state.append(sampler.sampling_algo.__getstate__())
        return self.position, self.current_index_of_dataset, self.current_index_of_sequence, self.current_is_sampling_positive_sample, dataset_samplers_state

    def load_state(self, state):
        self.position, self.current_index_of_dataset, self.current_index_of_sequence, self.current_is_sampling_positive_sample, dataset_samplers_state = state
        for index, dataset_sampler_state in enumerate(dataset_samplers_state):
            self.datasets_sampler[index].sampling_algo.__setstate__(dataset_sampler_state)
