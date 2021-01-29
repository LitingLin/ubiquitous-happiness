from Dataset.Base.factory import DatasetFactory
from Dataset.Base.Video.dataset import VideoDataset
from Dataset.Type.specialized_dataset import SpecializedVideoDatasetType
from Dataset.Base.Video.Filter.func import apply_filters_on_video_dataset_
from Dataset.MOT.Storage.MemoryMapped.dataset import MultipleObjectTrackingDataset_MemoryMapped

__all__ = ['MultipleObjectTrackingDatasetFactory']


class MultipleObjectTrackingDatasetFactory(DatasetFactory):
    def __init__(self, seeds: list):
        super(MultipleObjectTrackingDatasetFactory, self).__init__(seeds, VideoDataset,
                                                                   SpecializedVideoDatasetType.MultipleObjectTracking,
                                                                   apply_filters_on_video_dataset_,
                                                                   SpecializedVideoDatasetType.MultipleObjectTracking,
                                                                   MultipleObjectTrackingDataset_MemoryMapped)
