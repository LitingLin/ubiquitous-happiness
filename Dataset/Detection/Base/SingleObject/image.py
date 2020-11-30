import pathlib
from typing import List, Dict
from Dataset.Detection.Base.SingleObject.dataset import SingleObjectDetectionDataset


class SingleObjectDetectionDatasetImage:
    name: str
    image_path: pathlib.Path
    bounding_box: List
    category_id: int
    attributes: Dict


class SingleObjectDetectionDatasetImageViewer:
    dataset: SingleObjectDetectionDataset
    image: SingleObjectDetectionDatasetImage

    def __init__(self, dataset: SingleObjectDetectionDataset, image: SingleObjectDetectionDatasetImage):
        self.dataset = dataset
        self.image = image

    def getName(self):
        return self.image.name

    def getImagePath(self):
        return self.dataset.root_path.joinpath(self.image.image_path)

    def getBoundingBox(self):
        return self.image.bounding_box

    def getCategoryName(self):
        return self.dataset.category_names[self.image.category_id]

    def __iter__(self):
        yield from [self.getImagePath(), self.getBoundingBox(), self.getCategoryName()]
