from data.distributed.dataset import build_dataset_from_config_distributed_awareness
from data.tracking.sampler.SiamFC.sampler import SOTTrackingSiameseIterableDatasetSampler
from data.tracking.processor.siamfc.image_decoding import SiamFCImageDecodingProcessor
import numpy as np
import copy


def _customized_dataset_parameter_handler(datasets, parameters):
    number_of_datasets = len(datasets)
    datasets_parameters = [copy.deepcopy(parameters) for _ in range(number_of_datasets)]

    datasets_weight = np.array([len(dataset) for dataset in datasets], dtype=np.float_)
    datasets_weight = datasets_weight / datasets_weight.sum()

    assert 'NUM_USE' not in parameters, 'Unsupported flag'
    assert 'FRAME_RANGE' not in parameters, 'Unsupported flag'

    if 'Sampling' in parameters:
        sampling_parameters = parameters['Sampling']
        if 'weight' in sampling_parameters:
            for index, dataset_parameters in enumerate(datasets_parameters):
                dataset_parameters['Sampling']['weight'] = float(sampling_parameters['weight'] * datasets_weight[index])

    return datasets_parameters


def build_siamfc_sampling_dataset(data_config: dict, dataset_config_path: str, post_processor, seed: int):
    datasets, dataset_parameters = build_dataset_from_config_distributed_awareness(dataset_config_path, _customized_dataset_parameter_handler)

    samples_per_epoch = None
    negative_sample_ratio = 0

    if data_config is not None:
        if 'samples_per_epoch' in data_config:
            samples_per_epoch = data_config['samples_per_epoch']
        if 'repeat_times_per_epoch' in data_config:
            assert samples_per_epoch is None
            repeat_times_per_epoch = data_config['repeat_times_per_epoch']
            number_of_samples = sum(tuple(len(dataset) for dataset in datasets))
            samples_per_epoch = number_of_samples * repeat_times_per_epoch
        if 'neg_ratio' in data_config:
            negative_sample_ratio = data_config['neg_ratio']

    useful_dataset_parameters = [{}] * len(datasets)
    dataset_sampling_weights = []
    for useful_dataset_parameter, dataset_parameter in zip(useful_dataset_parameters, dataset_parameters):
        sampling_weight = 1
        if 'Sampling' in dataset_parameter:
            sampling_parameters = dataset_parameter['Sampling']
            if 'weight' in sampling_parameters:
                sampling_weight *= sampling_parameters['weight']
            if 'frame_range' in sampling_parameters:
                useful_dataset_parameter['frame_range'] = sampling_parameters['frame_range']
        dataset_sampling_weights.append(sampling_weight)

    dataset_sampling_weights = np.array(dataset_sampling_weights, dtype=np.float_)
    dataset_sampling_weights = dataset_sampling_weights / dataset_sampling_weights.sum()

    processor = SiamFCImageDecodingProcessor(post_processor)
    return SOTTrackingSiameseIterableDatasetSampler(datasets, samples_per_epoch, negative_sample_ratio, useful_dataset_parameters, dataset_sampling_weights, processor, seed)
