from typing import List, Dict
import pathlib


class SingleObjectDetectionDataset:
    name: str
    root_path: pathlib.Path
    images: List
    attributes: Dict

    category_names: List
    category_name_id_mapper: Dict

    def __init__(self):
        self.name = None
        self.root_path = None
        self.images = []
        self.category_names = []
        self.category_name_id_mapper = {}

    def __getitem__(self, index: int):
        return self.getImage(index)

    def __len__(self):
        return self.getNumberOfImages()

    def getRootPath(self):
        return str(self.root_path)

    def setRootPath(self, path: str):
        self.root_path = pathlib.Path(path)

    def getAttribure(self, name):
        return self.attributes[name]

    def getCategoryNameList(self):
        return self.category_names

    def getImage(self, index: int):
        from Dataset.Detection.Base.SingleObject.image import SingleObjectDetectionDatasetImageViewer
        return SingleObjectDetectionDatasetImageViewer(self, self.images[index])

    def getNumberOfImages(self):
        return len(self.images)

    def getConstructor(self):
        from Dataset.Detection.Base.SingleObject.constructor import SingleObjectDetectionConstructor
        return SingleObjectDetectionConstructor(self)
