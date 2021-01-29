from Dataset.Base.factory import DatasetFactory
from Dataset.Base.Video.dataset import VideoDataset
from Dataset.Type.specialized_dataset import SpecializedVideoDatasetType
from Dataset.Base.Video.Filter.func import apply_filters_on_video_dataset_
from Dataset.SOT.Storage.MemoryMapped.dataset import SingleObjectTrackingDataset_MemoryMapped

__all__ = ['SingleObjectTrackingDatasetFactory']


class SingleObjectTrackingDatasetFactory(DatasetFactory):
    def __init__(self, seeds: list):
        super(SingleObjectTrackingDatasetFactory, self).__init__(seeds, VideoDataset,
                                                                 SpecializedVideoDatasetType.SingleObjectTracking,
                                                                 apply_filters_on_video_dataset_,
                                                                 SpecializedVideoDatasetType.SingleObjectTracking,
                                                                 SingleObjectTrackingDataset_MemoryMapped)
