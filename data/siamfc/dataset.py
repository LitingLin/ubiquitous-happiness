# Copyright (c) SenseTime. All Rights Reserved.

import numpy as np
import torch
from torch.utils.data import Dataset

from Dataset.Builder.builder import build_datasets
import copy

from .curation import siamfc_z_curation
from NativeExtension import ImageDecoder
from torchvision.transforms import ToTensor
from .sampler import SOTDatasetSiamFCSampler, DetectionDatasetSiamFCSampler, MOTDatasetSiamFCSampler


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
    def __init__(self, dataset_config_path: str, samples_per_epoch, neg_ratio, post_processor=None, rng_seed: int=None):
        super(TrackingDataset, self).__init__()

        self.rng_engine = np.random
        if rng_seed is not None:
            self.rng_engine = np.random.default_rng(rng_seed)

        datasets = build_datasets(dataset_config_path)

        self.dataset_positioning = ConcateDatasetPositioning()
        self.pick = []
        self.datasets = []
        start_index = 0
        self.num = 0
        for dataset in datasets:
            self.dataset_positioning.register(len(dataset))
            if dataset.hasAttribute('NUM_USE'):
                num_use = dataset.getAttribute('NUM_USE')
            else:
                num_use = len(dataset)

            lists = list(range(start_index, len(dataset) + start_index))
            pick = []
            while len(pick) < num_use:
                self.rng_engine.shuffle(lists)
                pick += lists
            self.pick.extend(pick[:num_use])

            start_index += len(dataset)

            if dataset.hasAttribute('FRAME_RANGE'):
                frame_range = dataset.getAttribute('FRAME_RANGE')
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

        self.num = samples_per_epoch if samples_per_epoch > 0 else self.num
        self.pick = self.shuffle()
        self.image_decoder = ImageDecoder()
        self.to_tensor = ToTensor()
        self.post_processor = post_processor
        self.neg_ratio = neg_ratio

    def shuffle(self):
        pick = []

        m = 0
        while m < self.num:
            p = copy.copy(self.pick)
            self.rng_engine.shuffle(p)
            pick += p
            m = len(pick)
        print("shuffle done!")
        print("dataset length {}".format(self.num))
        return np.array(pick[:self.num], dtype=np.uint32)

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

            z_image = self.image_decoder.decode(z_path)
            x_image = self.image_decoder.decode(x_path)

            is_positive = False
        else:
            pair = dataset.get_positive_pair(index)
            if pair is None:
                while pair is None:
                    index = np.random.randint(0, len(dataset))
                    pair = dataset.get_positive_pair(index)

            if len(pair) == 2:
                z_path, z_bbox = pair
                z_image = self.image_decoder.decode(z_path)
                x_image = z_image
                x_bbox = z_bbox
            else:
                z_path, z_bbox, x_path, x_bbox = pair
                z_image = self.image_decoder.decode(z_path)
                x_image = self.image_decoder.decode(x_path)
            is_positive = True

        if self.post_processor is None:
            return z_image, z_bbox, x_image, x_bbox, is_positive
        return self.post_processor(z_image, z_bbox, x_image, x_bbox, is_positive)
