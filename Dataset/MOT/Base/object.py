from typing import List, Dict
from Utils.join_and_get_platform_style_path import join_and_get_platform_style_path


class MultipleObjectTrackingDataset_FrameObjectAttribute:
    bounding_box: List
    is_present: bool
    attributes: Dict


class MultipleObjectTrackingDataset_SequenceObjectAttribute:
    category_id: int
    frame_indices: List[int]
    attributes: Dict


class MultipleObjectTrackingDatasetSequenceObjectView:
    dataset: 'MultipleObjectTrackingDataset'
    sequence: 'MultipleObjectTrackingDatasetSequence'
    object_id: int

    def __init__(self, dataset: 'MultipleObjectTrackingDataset',
                 sequence: 'MultipleObjectTrackingDatasetSequence',
                 object_id: int):
        self.dataset = dataset
        self.sequence = sequence
        self.object_id = object_id
        self.object_attributes = self.sequence.object_id_attributes_mapper[self.object_id]

    def getCategoryName(self):
        return self.dataset.category_names[self.object_attributes.category_id]

    def getCategoryId(self):
        return self.object_attributes.category_id

    def getFrameIndices(self):
        return self.object_attributes.frame_indices

    def getAttribute(self, name: str):
        return self.object_attributes.attributes[name]

    def getAttributeNameList(self):
        return self.object_attributes.attributes.keys()

    def getAttributes(self):
        return self.object_attributes.attributes


class MultipleObjectTrackingDatasetFrameObjectView:
    dataset: 'MultipleObjectTrackingDataset'
    frame: 'MultipleObjectTrackingDatasetFrame'
    object_id: int
    frame_object_attributes: MultipleObjectTrackingDataset_FrameObjectAttribute
    object_attributes: MultipleObjectTrackingDataset_SequenceObjectAttribute

    def __init__(self, dataset: 'MultipleObjectTrackingDataset', frame: 'MultipleObjectTrackingDatasetFrame',
                 object_id: int,
                 object_attributes: MultipleObjectTrackingDataset_SequenceObjectAttribute,
                 frame_object_information: MultipleObjectTrackingDataset_FrameObjectAttribute):
        self.dataset = dataset
        self.frame = frame
        self.object_id = object_id
        self.object_attributes = object_attributes
        self.frame_object_attributes = frame_object_information

    def getCategoryName(self):
        return self.dataset.category_names[self.object_attributes.category_id]

    def getCategoryId(self):
        return self.object_attributes.category_id

    def getImagePath(self):
        return join_and_get_platform_style_path(self.dataset.root_path, self.frame.image_path)

    def getBoundingBox(self):
        return self.frame_object_attributes.bounding_box

    def hasAttributeIsPresent(self):
        return self.frame_object_attributes.is_present is not None

    def getAttributeIsPresent(self):
        return self.frame_object_attributes.is_present

    def getObjectId(self):
        return self.object_id

    def getAttribute(self, name: str):
        return self.frame_object_attributes.attributes[name]

    def getAttributeNameList(self):
        return self.frame_object_attributes.attributes.keys()

    def getAttributes(self):
        return self.frame_object_attributes.attributes
