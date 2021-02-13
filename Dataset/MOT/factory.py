from Dataset.Base.factory import DatasetFactory
from Dataset.Base.Video.dataset import VideoDataset
from Dataset.Type.specialized_dataset import SpecializedVideoDatasetType
from Dataset.Base.Video.Filter.func import apply_filters_on_video_dataset_
from Dataset.MOT.Storage.MemoryMapped.dataset import MultipleObjectTrackingDataset_MemoryMapped
from Dataset.Type.bounding_box_format import BoundingBoxFormat
from typing import List

__all__ = ['MultipleObjectTrackingDatasetFactory']


class MultipleObjectTrackingDatasetFactory(DatasetFactory):
    def __init__(self, seeds: list):
        super(MultipleObjectTrackingDatasetFactory, self).__init__(seeds, VideoDataset,
                                                                   SpecializedVideoDatasetType.MultipleObjectTracking,
                                                                   apply_filters_on_video_dataset_,
                                                                   SpecializedVideoDatasetType.MultipleObjectTracking,
                                                                   MultipleObjectTrackingDataset_MemoryMapped)

    def construct(self, filters: list=None, cache_base_format: bool=False, dump_human_readable: bool=False, bounding_box_format: BoundingBoxFormat = BoundingBoxFormat.XYWH) -> List[MultipleObjectTrackingDataset_MemoryMapped]:
        return super(MultipleObjectTrackingDatasetFactory, self).construct(filters, cache_base_format, dump_human_readable, bounding_box_format)

    def construct_base_interface(self, filters=None, make_cache=False, dump_human_readable=False) -> List[VideoDataset]:
        return super(MultipleObjectTrackingDatasetFactory, self).construct_base_interface(filters, make_cache, dump_human_readable)
