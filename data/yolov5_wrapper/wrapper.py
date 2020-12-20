from Dataset.Detection.Base.MemoryMapped.dataset import DetectionDataset_MemoryMapped
from torch.utils.data.dataset import Dataset
from ._pipeline import image_loading_pipeline
from ._mosaic_pipeline import mosaic_image_loading_pipeline
import random


class YoloV5Dataset(Dataset):
    def __init__(self, dataset: DetectionDataset_MemoryMapped, target_image_size: int=640, do_augmentation: bool=False, aug_config: dict=None, rectangular_output:bool=False, batch_size=16, stride=32, pad=0.0):
        self.dataset = dataset
        self.target_image_size = target_image_size
        self.do_augmentation = do_augmentation
        self.do_mosaic = do_augmentation and not rectangular_output
        self.aug_config = aug_config
        self.rectangular_output = rectangular_output
        self.batch_size = batch_size
        self.stride = stride
        self.pad = pad

    def __getitem__(self, index: int):
        mosaic = self.do_mosaic and random.random() < self.aug_config['mosaic']
        if mosaic:
            return mosaic_image_loading_pipeline(self.dataset, index, self.target_image_size, self.do_augmentation, self.aug_config)
        else:
            return image_loading_pipeline(self.dataset, index, self.target_image_size, self.do_augmentation, self.aug_config, self.rectangular_output, self.batch_size, self.stride, self.pad)

    def __len__(self):
        return len(self.dataset)
