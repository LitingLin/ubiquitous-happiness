from Dataset.Base.factory import DatasetFactory
from Dataset.Base.Image.dataset import ImageDataset
from Dataset.Type.specialized_dataset import SpecializedImageDatasetType
from Dataset.Base.Image.Filter.func import apply_filters_on_image_dataset_
from Dataset.DET.Storage.MemoryMapped.dataset import DetectionDataset_MemoryMapped

__all__ = ['DetectionDatasetFactory']


class DetectionDatasetFactory(DatasetFactory):
    def __init__(self, seeds: list):
        super(DetectionDatasetFactory, self).__init__(seeds, ImageDataset,
                                                      SpecializedImageDatasetType.Detection,
                                                      apply_filters_on_image_dataset_,
                                                      SpecializedImageDatasetType.Detection,
                                                      DetectionDataset_MemoryMapped)
