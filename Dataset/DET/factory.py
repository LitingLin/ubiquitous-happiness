from Dataset.Base.factory import DatasetFactory
from Dataset.Base.Image.dataset import ImageDataset
from Dataset.Type.specialized_dataset import SpecializedImageDatasetType
from Dataset.Base.Image.Filter.func import apply_filters_on_image_dataset_
from Dataset.DET.Storage.MemoryMapped.dataset import DetectionDataset_MemoryMapped
from typing import List

__all__ = ['DetectionDatasetFactory']


class DetectionDatasetFactory(DatasetFactory):
    def __init__(self, seeds: list):
        super(DetectionDatasetFactory, self).__init__(seeds, ImageDataset,
                                                      SpecializedImageDatasetType.Detection,
                                                      apply_filters_on_image_dataset_,
                                                      SpecializedImageDatasetType.Detection,
                                                      DetectionDataset_MemoryMapped)

    def construct(self, filters: list=None, cache_base_format: bool=True, dump_human_readable: bool=False) -> List[DetectionDataset_MemoryMapped]:
        return super(DetectionDatasetFactory, self).construct(filters, cache_base_format, dump_human_readable)

    def construct_base_interface(self, filters=None, make_cache=False, dump_human_readable=False) -> List[ImageDataset]:
        return super(DetectionDatasetFactory, self).construct_base_interface(filters, make_cache, dump_human_readable)
