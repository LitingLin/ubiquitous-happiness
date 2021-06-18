from Dataset.Base.factory import DatasetFactory
from Dataset.Base.Image.dataset import ImageDataset
from Dataset.Type.specialized_dataset import SpecializedImageDatasetType
from Dataset.Base.Image.Filter.func import apply_filters_on_image_dataset_
from Dataset.CLS.Storage.MemoryMapped.dataset import ImageClassificationDataset_MemoryMapped
from typing import List

__all__ = ['ImageClassificationDatasetFactory']


class ImageClassificationDatasetFactory(DatasetFactory):
    def __init__(self, seeds: list):
        super(ImageClassificationDatasetFactory, self).__init__(seeds, ImageDataset,
                                                                SpecializedImageDatasetType.Classification,
                                                                apply_filters_on_image_dataset_,
                                                                SpecializedImageDatasetType.Classification,
                                                                ImageClassificationDataset_MemoryMapped)

    def construct(self, filters: list = None, cache_base_format: bool = True, dump_human_readable: bool = False) -> List[ImageClassificationDataset_MemoryMapped]:
        return super(ImageClassificationDatasetFactory, self).construct(filters, cache_base_format, dump_human_readable)

    def construct_base_interface(self, filters=None, make_cache=False, dump_human_readable=False) -> List[ImageDataset]:
        return super(ImageClassificationDatasetFactory, self).construct_base_interface(filters, make_cache,
                                                                                       dump_human_readable)
