# Copyright (c) SenseTime. All Rights Reserved.

import numpy as np
import torch
from torch.utils.data import Dataset

from Dataset.Builder.builder import build_datasets
import copy

from Dataset.Detection.Base.MemoryMapped.dataset import DetectionDataset_MemoryMapped
from Dataset.MOT.Base.dataset import MultipleObjectTrackingDataset
from Dataset.SOT.Base.MemoryMapped.dataset import SingleObjectTrackingDataset_MemoryMapped
from .curation import siamfc_z_curation
from NativeExtension import ImageDecoder
from torchvision.transforms import ToTensor
import cv2


class DatasetWrapper:
    @staticmethod
    def _get_data(frame):
        image_path = frame.getImagePath()
        assert isinstance(image_path, str)

        bounding_box = frame.getBoundingBox()
        assert bounding_box is not None
        assert len(bounding_box) == 4
        return image_path, bounding_box


class DetectionDatasetWrapper:
    def __init__(self, dataset: DetectionDataset_MemoryMapped):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def get_positive_pair(self, index):
        image = self.dataset[index]
        object_index = np.random.randint(0, len(image))
        object_ = image[object_index]
        image_path = image.getImagePath()
        assert isinstance(image_path, str)
        bounding_box = object_.getBoundingBox()
        assert bounding_box is not None
        assert len(bounding_box) == 4

        return (image_path, bounding_box), (image_path, bounding_box)

    def get_random_target(self, index=-1):
        if index == -1:
            index = np.random.randint(0, len(self.dataset))
        image = self.dataset[index]
        object_index = np.random.randint(0, len(image))
        object_ = image[object_index]
        image_path = image.getImagePath()
        assert isinstance(image_path, str)
        bounding_box = object_.getBoundingBox()
        assert bounding_box is not None
        assert len(bounding_box) == 4

        return image_path, bounding_box


class SOTDatasetWrapper(DatasetWrapper):
    def __init__(self, dataset: SingleObjectTrackingDataset_MemoryMapped, frame_range: int):
        self.dataset = dataset
        self.frame_range = frame_range

    def __len__(self):
        return len(self.dataset)

    def get_positive_pair(self, index: int):
        sequence = self.dataset[index]

        template_frame_index = np.random.randint(0, len(sequence))

        left = max(template_frame_index - self.frame_range, 0)
        right = min(template_frame_index + self.frame_range, len(sequence) - 1) + 1

        search_frame_index = np.random.randint(left, right)

        template_frame = sequence[template_frame_index]
        search_frame = sequence[search_frame_index]
        return DatasetWrapper._get_data(template_frame), DatasetWrapper._get_data(search_frame)

    def get_random_target(self, index: int = -1):
        if index == -1:
            index = np.random.randint(0, len(self.dataset))
        sequence = self.dataset[index]

        frame_index = np.random.randint(0, len(sequence))
        frame = sequence[frame_index]

        return DatasetWrapper._get_data(frame)


class MOTDatasetWrapper(DatasetWrapper):
    def __init__(self, dataset: MultipleObjectTrackingDataset, frame_range: int):
        self.dataset = dataset
        self.frame_range = frame_range

    def __len__(self):
        return len(self.dataset)

    def get_positive_pair(self, index):
        sequence = self.dataset[index]

        track_index = np.random.randint(0, sequence.getNumberOfTracks())
        track = sequence.getTrackView(track_index)

        template_track_index = np.random.randint(0, len(track))
        left = max(template_track_index - self.frame_range, 0)
        right = min(template_track_index + self.frame_range, len(track) - 1) + 1

        search_track_index = np.random.randint(left, right)

        template_frame = track[template_track_index]
        search_frame = track[search_track_index]
        return MOTDatasetWrapper._get_data(template_frame), MOTDatasetWrapper._get_data(search_frame)

    def get_random_target(self, index=-1):
        if index == -1:
            index = np.random.randint(0, len(self.dataset))
        sequence = self.dataset[index]

        track_index = np.random.randint(0, sequence.getNumberOfTracks())
        track = sequence.getTrackView(track_index)
        frame_index = np.random.randint(0, len(track))
        frame = track[frame_index]

        return MOTDatasetWrapper._get_data(frame)


class ImageSizeLimiter:
    def __init__(self, max_size):
        self.max_size = max_size

    def __call__(self, image, bounding_box):
        h, w = image.shape[0:2]
        ratio = min([self.max_size / h, self.max_size / w, 1])
        if ratio == 1:
            return image, bounding_box
        else:
            dst_size = (int(round(ratio * w)), int(round(ratio * h)))
            bounding_box = [int(round(v * ratio)) for v in bounding_box]
            image = cv2.resize(image, dst_size)
            return image, bounding_box


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


class TrkDataset(Dataset):
    def __init__(self, dataset_config_path: str, samples_per_epoch, exemplar_size, max_size):
        super(TrkDataset, self).__init__()

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
                np.random.shuffle(lists)
                pick += lists
            self.pick.extend(pick[:num_use])

            start_index += len(dataset)

            if dataset.hasAttribute('FRAME_RANGE'):
                frame_range = dataset.getAttribute('FRAME_RANGE')
            else:
                frame_range = 100

            class_name: str = dataset.__class__.__name__
            if class_name.startswith('Detection'):
                self.datasets.append(DetectionDatasetWrapper(dataset))
            elif class_name.startswith('Single'):
                self.datasets.append(SOTDatasetWrapper(dataset, frame_range))
            elif class_name.startswith('Multiple'):
                self.datasets.append(MOTDatasetWrapper(dataset, frame_range))
            else:
                raise Exception('Unknown dataset type')
            self.num += num_use

        self.num = samples_per_epoch if samples_per_epoch > 0 else self.num
        self.pick = self.shuffle()
        self.exemplar_size = exemplar_size
        self.image_decoder = ImageDecoder()
        self.to_tensor = ToTensor()
        self.image_size_limiter = ImageSizeLimiter(max_size)

    def shuffle(self):
        pick = []

        m = 0
        while m < self.num:
            p = copy.copy(self.pick)
            np.random.shuffle(p)
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

        template, search = dataset.get_positive_pair(index)

        template_image_path, template_bounding_box = template
        search_image_path, search_bounding_box = search

        template_image = self.image_decoder.decode(template_image_path)
        search_image: np.ndarray = self.image_decoder.decode(search_image_path)

        search_image, search_bounding_box = self.image_size_limiter(search_image, search_bounding_box)

        template_image = siamfc_z_curation(template_image, template_bounding_box, 0.5, self.exemplar_size)

        search_bounding_box = [search_bounding_box[0] + search_bounding_box[2] / 2,
                               search_bounding_box[1] + search_bounding_box[3] / 2,
                               search_bounding_box[2], search_bounding_box[3]]

        search_image_h, search_image_w = search_image.shape[0:2]
        search_bounding_box = [search_bounding_box[0] / search_image_w, search_bounding_box[1] / search_image_h,
                               search_bounding_box[2] / search_image_w, search_bounding_box[3] / search_image_h]

        return self.to_tensor(template_image), self.to_tensor(search_image), {
            'bbox': torch.tensor(search_bounding_box, dtype=torch.float32),
            'size': torch.tensor((search_image_w, search_image_h), dtype=torch.int),
            'z_path': template_image_path, 'x_path': search_image_path}
