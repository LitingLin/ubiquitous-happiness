from typing import List, Dict, Any

from Dataset.MOT.Base.dataset import MultipleObjectTrackingDataset
from Dataset.MOT.Base.frame import MultipleObjectTrackingDatasetFrame
from Dataset.MOT.Base.object import MultipleObjectTrackingDataset_SequenceObjectAttribute


class MultipleObjectTrackingDatasetSequence:
    name: str
    frames: List[MultipleObjectTrackingDatasetFrame]
    object_ids: List[int]
    fps: float
    object_id_attributes_mapper: Dict[int, MultipleObjectTrackingDataset_SequenceObjectAttribute]
    attributes: Dict[str, Any]

    def __init__(self):
        self.name = None
        self.frames = []
        self.object_ids = []
        self.object_id_attributes_mapper = {}
        self.attributes = {}
        self.fps = None


class MultipleObjectTrackingDatasetSequenceView:
    def __init__(self, dataset: MultipleObjectTrackingDataset, sequence: MultipleObjectTrackingDatasetSequence):
        self.dataset = dataset
        self.sequence = sequence

    def getName(self):
        return self.sequence.name

    def getNumberOfFrames(self):
        return len(self.sequence.frames)

    def getObjectIds(self):
        return self.sequence.object_id_attributes_mapper.keys()

    def getAttribute(self, name: str):
        return self.sequence.attributes[name]

    def hasAttribute(self, name: str):
        return name in self.sequence.attributes

    def getFrame(self, index: int):
        from .frame import MultipleObjectTrackingDatasetFrameView
        return MultipleObjectTrackingDatasetFrameView(self.dataset, self.sequence.frames[index],
                                                      self.sequence.object_id_attributes_mapper)

    def getObjectInformation(self, object_id: int):
        from .object import MultipleObjectTrackingDatasetSequenceObjectView
        return MultipleObjectTrackingDatasetSequenceObjectView(self.dataset, self.sequence, object_id)

    def __len__(self):
        return self.getNumberOfFrames()

    def __getitem__(self, index: int):
        return self.getFrame(index)

    def getNumberOfTracks(self):
        return len(self.sequence.object_ids)

    def getTrackView(self, index: int):
        from .track import MultipleObjectTrackingDatasetSequenceTrackView
        return MultipleObjectTrackingDatasetSequenceTrackView(self.dataset, self.sequence,
                                                              self.sequence.object_ids[index])

    def getTrackIterator(self):
        from .track import MultipleObjectTrackingDatasetSequenceTrackIterator
        return MultipleObjectTrackingDatasetSequenceTrackIterator(self.dataset, self.sequence)

    def getFPS(self):
        return self.sequence.fps
