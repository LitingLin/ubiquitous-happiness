from typing import List, Dict
from Utils.join_and_get_platform_style_path import join_and_get_platform_style_path


class DetectionDatasetImage:
    name: str
    image_path: str
    attributes: Dict
    objects: List
    size: List # [width, height]


class DetectionDatasetImageViewer:
    def __init__(self, dataset, image: DetectionDatasetImage):
        self.dataset = dataset
        self.image = image

    def __len__(self):
        return self.getNumberObjects()

    def getName(self):
        return self.image.name

    def getImagePath(self):
        return join_and_get_platform_style_path(self.dataset.root_path, self.image.image_path)

    def getAttribute(self, name: str):
        return self.image.attributes[name]

    def hasAttribute(self, name: str):
        return name in self.image.attributes

    def getNumberObjects(self):
        return len(self.image.objects)

    def getImageSize(self):
        return self.image.size

    def getObject(self, index: int):
        from Dataset.Detection.Base.object import DetectionDatasetObjectViewer
        return DetectionDatasetObjectViewer(self.dataset, self.image, self.image.objects[index])

    def __getitem__(self, index: int):
        return self.getObject(index)

    def hasAttributeCategory(self):
        return self.dataset.hasAttributeCategory()

    def hasAttibuteIsPresent(self):
        return self.dataset.hasAttibuteIsPresent()
