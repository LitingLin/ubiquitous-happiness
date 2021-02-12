from Dataset.Base.factory import DatasetFactory
from Dataset.Base.Video.dataset import VideoDataset
from Dataset.Type.specialized_dataset import SpecializedVideoDatasetType
from Dataset.Base.Video.Filter.func import apply_filters_on_video_dataset_
from Dataset.SOT.Storage.MemoryMapped.dataset import SingleObjectTrackingDataset_MemoryMapped
from Dataset.Type.bounding_box_format import BoundingBoxFormat
from typing import List

__all__ = ['SingleObjectTrackingDatasetFactory']


class SingleObjectTrackingDatasetFactory(DatasetFactory):
    def __init__(self, seeds: list):
        super(SingleObjectTrackingDatasetFactory, self).__init__(seeds, VideoDataset,
                                                                 SpecializedVideoDatasetType.SingleObjectTracking,
                                                                 apply_filters_on_video_dataset_,
                                                                 SpecializedVideoDatasetType.SingleObjectTracking,
                                                                 SingleObjectTrackingDataset_MemoryMapped)

    def construct(self, filters: list=None, cache_base_format: bool=False, dump_human_readable: bool=False, bounding_box_format: BoundingBoxFormat = BoundingBoxFormat.XYWH) -> List[SingleObjectTrackingDataset_MemoryMapped]:
        return super(SingleObjectTrackingDatasetFactory, self).construct(filters, cache_base_format, dump_human_readable, bounding_box_format)

    def construct_base_interface(self, filters=None, make_cache=False, dump_human_readable=False) -> List[VideoDataset]:
        return super(SingleObjectTrackingDatasetFactory, self).construct_base_interface(filters, make_cache, dump_human_readable)
