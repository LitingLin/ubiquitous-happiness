from typing import List, Dict
from Utils.join_and_get_platform_style_path import join_and_get_platform_style_path


class SingleObjectTrackingDatasetFrame:
    image_path: str
    bounding_box: List
    is_present: bool
    attributes: Dict
    size: List  # [width, height]


class SingleObjectTrackingDatasetFrameViewer:
    def __init__(self, dataset, sequence, frame: SingleObjectTrackingDatasetFrame):
        self.dataset = dataset
        self.sequence = sequence
        self.frame = frame

    def getImagePath(self):
        return join_and_get_platform_style_path(self.dataset.root_path, self.frame.image_path)

    def hasAttributes(self):
        return hasattr(self.frame, 'bounding_box')

    def getBoundingBox(self):
        return self.frame.bounding_box

    def getAttributeIsPresent(self):
        return self.frame.is_present

    def getClassName(self):
        return self.dataset.class_label_names[self.sequence.class_label]

    def getAttribute(self, name):
        return self.frame.attributes[name]

    def __iter__(self):
        yield from [self.getImagePath(), self.getBoundingBox(), self.getAttributeIsPresent()]
