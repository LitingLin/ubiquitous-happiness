# Copyright (c) SenseTime. All Rights Reserved.
import copy

import numpy as np
from torch.utils.data import Dataset
from data.distributed.dataset import build_dataset_from_config_distributed_awareness
from miscellanies.torch.distributed import is_main_process
from torchvision.transforms import ToTensor
import torchvision.io
from data.siamfc.sampler import SOTDatasetSiamFCSampler, DetectionDatasetSiamFCSampler, MOTDatasetSiamFCSampler


def _decode_image(path):
    return torchvision.io.read_image(path, torchvision.io.image.ImageReadMode.RGB)


class ConcateDatasetPositioning:
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


class TrackingDataset(Dataset):
    def __init__(self, datasets: list, datasets_parameter, samples_per_epoch, repeat_times_per_epoch, neg_ratio, post_processor=None, rng_seed: int=None):
        super(TrackingDataset, self).__init__()

        self.rng_engine = np.random
        if rng_seed is not None:
            self.rng_engine = np.random.default_rng(rng_seed)

        self.dataset_positioning = ConcateDatasetPositioning()
        self.pick_seed = []
        self.datasets = []
        start_index = 0
        self.num = 0
        if is_main_process():
            print('Building dataset indices...', end=' ')
        for dataset, dataset_parameter in zip(datasets, datasets_parameter):
            self.dataset_positioning.register(len(dataset))
            if 'NUM_USE' in dataset_parameter:
                num_use = dataset_parameter['NUM_USE']
            else:
                num_use = len(dataset)

            dataset_indices = np.arange(start_index, len(dataset) + start_index)
            used = 0
            picks = []
            while used < num_use:
                dataset_indices_ = dataset_indices.copy()
                self.rng_engine.shuffle(dataset_indices_)
                used += len(dataset_indices_)
                picks.append(dataset_indices_)
            picks = np.concatenate(picks)
            picks = picks[: num_use]
            self.pick_seed.append(picks)

            start_index += len(dataset)

            if 'FRAME_RANGE' in dataset_parameter:
                frame_range = dataset_parameter['FRAME_RANGE']
            else:
                frame_range = 100

            class_name: str = dataset.__class__.__name__
            if class_name.startswith('Detection'):
                self.datasets.append(DetectionDatasetSiamFCSampler(dataset))
            elif class_name.startswith('Single'):
                self.datasets.append(SOTDatasetSiamFCSampler(dataset, frame_range))
            elif class_name.startswith('Multiple'):
                self.datasets.append(MOTDatasetSiamFCSampler(dataset, frame_range))
            else:
                raise Exception('Unknown dataset type')
            self.num += num_use
        self.pick_seed = np.concatenate(self.pick_seed)

        self.num = samples_per_epoch if samples_per_epoch is not None and samples_per_epoch > 0 else self.num
        if repeat_times_per_epoch is not None and repeat_times_per_epoch > 0:
            self.num *= repeat_times_per_epoch
        if is_main_process():
            print('done.')
        self.generate_shuffled_picks()
        self.to_tensor = ToTensor()
        self.post_processor = post_processor
        self.neg_ratio = neg_ratio
        self.rng_seed = rng_seed

    def set_epoch(self, epoch):
        self.generate_shuffled_picks(epoch)

    def generate_shuffled_picks(self, seed=None):
        if is_main_process():
            print('Shuffling datasets...', end=' ')
        if seed is not None:
            if self.rng_seed is not None:
                seed += self.rng_seed
            rng_engine = np.random.default_rng(seed)
        else:
            rng_engine = self.rng_engine
        picks = []
        picked = 0
        while picked < self.num:
            p = self.pick_seed.copy()
            rng_engine.shuffle(p)
            picks.append(p)
            picked += len(self.pick_seed)
        picks = np.concatenate(picks)
        if is_main_process():
            print('done.')
        self.pick = picks[:self.num]

    def __len__(self):
        return self.num

    def __getitem__(self, index):
        index = self.pick[index]
        index_of_dataset, index = self.dataset_positioning(index)
        dataset = self.datasets[index_of_dataset]

        if self.neg_ratio > 0 and self.neg_ratio > np.random.random():
            # neg
            z = dataset.get_random_target()
            if z is None:
                while z is None:
                    z = np.random.choice(self.datasets).get_random_target()
            x = None
            while x is None:
                x = np.random.choice(self.datasets).get_random_target()
            z_path, z_bbox = z
            x_path, x_bbox = x

            z_image = _decode_image(z_path)
            x_image = _decode_image(x_path)

            is_positive = False
        else:
            pair = dataset.get_positive_pair(index)
            if pair is None:
                while pair is None:
                    index = np.random.randint(0, len(dataset))
                    pair = dataset.get_positive_pair(index)

            if len(pair) == 2:
                z_path, z_bbox = pair
                z_image = _decode_image(z_path)
                x_image = z_image
                x_bbox = z_bbox
            else:
                z_path, z_bbox, x_path, x_bbox = pair
                z_image = _decode_image(z_path)
                x_image = _decode_image(x_path)
            is_positive = True

        if self.post_processor is None:
            return z_image, z_bbox, x_image, x_bbox, is_positive
        return self.post_processor(z_image, z_bbox, x_image, x_bbox, is_positive)


def _customized_dataset_parameter_handler(datasets, parameters):
    number_of_datasets = len(datasets)
    datasets_parameters = [copy.deepcopy(parameters) for _ in range(number_of_datasets)]

    datasets_weight = np.array([len(dataset) for dataset in datasets], dtype=np.float_)
    datasets_weight = datasets_weight / datasets_weight.sum()

    if 'NUM_USE' in parameters:
        num_use = parameters['NUM_USE']
        for index in range(number_of_datasets):
            datasets_parameters[index]['NUM_USE'] = int((num_use * datasets_weight[index]).round())

    return datasets_parameters


def _build_tracking_dataset(data_config, dataset_config_path, post_processor, rng_seed):
    raw_datasets, dataset_parameters = build_dataset_from_config_distributed_awareness(dataset_config_path, _customized_dataset_parameter_handler)
    # default values
    samples_per_epoch = None
    repeat_times_per_epoch = None
    neg_ratio = 0.

    if data_config is not None:
        if 'samples_per_epoch' in data_config:
            samples_per_epoch = data_config['samples_per_epoch']
        if 'repeat_times_per_epoch' in data_config:
            repeat_times_per_epoch = data_config['repeat_times_per_epoch']
        if 'negative_sample_ratio' in data_config:
            neg_ratio = data_config['negative_sample_ratio']

    dataset = TrackingDataset(raw_datasets, dataset_parameters, samples_per_epoch, repeat_times_per_epoch, neg_ratio, post_processor, rng_seed)
    return dataset


def build_tracking_dataset(train_config: dict, train_dataset_config_path: str, val_dataset_config_path: str, train_post_processor, val_post_processor):
    train_data_config = train_config['data']['sampler']['train']
    train_dataset = _build_tracking_dataset(train_data_config, train_dataset_config_path, train_post_processor, 33)

    val_data_config = train_config['data']['sampler']['val']
    val_dataset = _build_tracking_dataset(val_data_config, val_dataset_config_path, val_post_processor, 44)

    return train_dataset, val_dataset
