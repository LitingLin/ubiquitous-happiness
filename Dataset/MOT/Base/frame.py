from typing import Dict, Any, List
from Dataset.MOT.Base.object import MultipleObjectTrackingDataset_FrameObjectAttribute, MultipleObjectTrackingDataset_SequenceObjectAttribute
from Dataset.MOT.Base.dataset import MultipleObjectTrackingDataset
from Utils.join_and_get_platform_style_path import join_and_get_platform_style_path


class MultipleObjectTrackingDatasetFrame:
    objects: Dict[int, MultipleObjectTrackingDataset_FrameObjectAttribute]
    image_path: str
    attributes: Dict[str, Any]
    size: List  # [width, height]


class MultipleObjectTrackingDatasetSequenceFrameObjectIterator:
    dataset: MultipleObjectTrackingDataset
    frame: MultipleObjectTrackingDatasetFrame
    object_information: Dict[int, MultipleObjectTrackingDataset_SequenceObjectAttribute]

    def __init__(self, dataset: MultipleObjectTrackingDataset, frame: MultipleObjectTrackingDatasetFrame,
                 object_information: Dict[int, MultipleObjectTrackingDataset_SequenceObjectAttribute]):
        self.dataset = dataset
        self.frame = frame
        self.object_information = object_information
        self.iter = iter(self.frame.objects.items())

    def __next__(self):
        object_id, frame_object_information = next(self.iter)
        from .object import MultipleObjectTrackingDatasetFrameObjectView
        return MultipleObjectTrackingDatasetFrameObjectView(self.dataset, self.frame,
                                                            object_id, self.object_information[object_id],
                                                            frame_object_information)


class MultipleObjectTrackingDatasetFrameView:
    dataset: MultipleObjectTrackingDataset
    frame: MultipleObjectTrackingDatasetFrame
    object_attributes: Dict[int, MultipleObjectTrackingDataset_SequenceObjectAttribute]

    def __init__(self, dataset: MultipleObjectTrackingDataset, frame: MultipleObjectTrackingDatasetFrame,
                 object_attributes: Dict[int, MultipleObjectTrackingDataset_SequenceObjectAttribute]):
        self.dataset = dataset
        self.frame = frame
        self.object_attributes = object_attributes

    def getDataset(self):
        return self.dataset

    def getObjectIds(self):
        return self.frame.objects.keys()

    def __iter__(self):
        return MultipleObjectTrackingDatasetSequenceFrameObjectIterator(self.dataset, self.frame,
                                                                        self.object_attributes)

    def getObjectById(self, object_id: int):
        from .object import MultipleObjectTrackingDatasetFrameObjectView
        return MultipleObjectTrackingDatasetFrameObjectView(self.dataset, self.frame,
                                                            object_id, self.object_attributes[object_id],
                                                            self.frame.objects[object_id])

    def getImagePath(self):
        return join_and_get_platform_style_path(self.dataset.root_path, self.frame.image_path)

    def getNumberOfObjects(self):
        return len(self.frame.objects)

    def getAttribute(self, name):
        return self.frame.attributes[name]

    def getAttributeNameList(self):
        return self.frame.attributes.keys()

    def getAttributes(self):
        return self.frame.attributes
