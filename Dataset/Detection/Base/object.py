from typing import List, Dict
from Dataset.Detection.Base.dataset import DetectionDataset
from Dataset.Detection.Base.image import DetectionDatasetImage
from Utils.join_and_get_platform_style_path import join_and_get_platform_style_path


class DetectionDatasetObject:
    bounding_box: List
    category_id: int
    attributes: Dict
    is_present: bool


class DetectionDatasetObjectViewer:
    def __init__(self, dataset: DetectionDataset, image: DetectionDatasetImage, object_: DetectionDatasetObject):
        self.dataset = dataset
        self.image = image
        self.object_ = object_

    def getName(self):
        return self.image.name + '-' + self.getCategoryName()

    def getImagePath(self):
        return join_and_get_platform_style_path(self.dataset.root_path, self.image.image_path)

    def getBoundingBox(self):
        return self.object_.bounding_box

    def getCategoryName(self):
        return self.dataset.category_names[self.object_.category_id]

    def getAttribute(self, name):
        return self.object_.attributes[name]

    def hasAttributeCategory(self):
        return self.dataset.hasAttributeCategory()

    def hasAttributeIsPresent(self):
        return self.dataset.hasAttibuteIsPresent()

    def getAttributeIsPresent(self):
        return self.object_.is_present
