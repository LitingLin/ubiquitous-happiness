from Dataset.DataSplit import DataSplit
from typing import List, Dict

class DetectionDataset:
    name: str
    images: List
    root_path: str
    attributes: Dict

    category_names: List
    category_name_id_mapper: Dict

    def __init__(self):
        self.structure_version = 3
        self.data_version = None
        self.filters = []
        self.name = None
        self.data_split = DataSplit.Full
        self.root_path = None
        self.images = []
        self.attributes = {}

        self.category_names = []
        self.category_name_id_mapper = {}

    def getCategoryNameList(self):
        return self.category_names

    def getNumberOfCategories(self):
        return len(self.category_names)

    def getCategoryName(self, id_: int):
        return self.category_names[id_]

    def getDataSplit(self):
        return self.data_split

    def setRootPath(self, root_path: str):
        assert isinstance(root_path, str)
        self.root_path = root_path

    def isSingleObject(self):
        return self.attributes['single_object']

    def getName(self):
        return self.name

    def getRootPath(self):
        return self.root_path

    def __len__(self):
        return len(self.images)

    def getNumberOfImages(self):
        return len(self.images)

    def getImage(self, index: int):
        from Dataset.Detection.Base.image import DetectionDatasetImageViewer
        return DetectionDatasetImageViewer(self, self.images[index])

    def getAttribute(self, name):
        return self.attributes[name]

    def hasAttribute(self, name):
        return name in self.attributes

    def getAttributes(self):
        return self.attributes

    def __getitem__(self, index: int):
        return self.getImage(index)

    def getConstructor(self):
        from Dataset.Detection.Base.constructor import DetectionDatasetConstructor
        return DetectionDatasetConstructor(self)

    def getFlattenView(self):
        from .flatten_view import DetectionDatasetFlattenView
        return DetectionDatasetFlattenView(self)

    def hasAttributeCategory(self):
        return self.attributes['has_object_category_attr']

    def hasAttibuteIsPresent(self):
        return self.attributes['has_is_present_attr']

    def getModifier(self):
        from .modifier import DetectionDatasetModifier
        return DetectionDatasetModifier(self)
