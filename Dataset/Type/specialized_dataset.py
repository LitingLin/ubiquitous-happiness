from enum import Enum, auto


class SpecializedImageDatasetType(Enum):
    Detection = auto()


class SpecializedVideoDatasetType(Enum):
    SingleObjectTracking = auto()
    MultipleObjectTracking = auto()
