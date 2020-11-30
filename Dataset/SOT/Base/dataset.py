from typing import List, Dict
from Dataset.DataSplit import DataSplit


class SingleObjectTrackingDataset:
    sequences: List['SingleObjectTrackingDatasetSequence']
    root_path: str
    category_names: List[str]
    category_name_id_mapper: Dict[str, int]
    name: str
    attributes: Dict

    def __init__(self):
        self.data_version = None
        self.structure_version = 5
        self.data_split = DataSplit.Full
        self.filters = []
        self.sequences = []
        self.root_path = None
        self.category_name_id_mapper = {}
        self.category_names = []
        self.name = None
        self.attributes = {}

    def setRootPath(self, root_path: str):
        assert isinstance(root_path, str)
        self.root_path = root_path

    def getRootPath(self):
        return self.root_path

    def getName(self):
        return self.name

    def getCategoryNameList(self):
        return self.category_names

    def getNumberOfCategories(self):
        return len(self.category_names)

    def getCategoryName(self, id_: int):
        return self.category_names[id_]

    def getView(self):
        from .view import SingleObjectTrackingDatasetView
        return SingleObjectTrackingDatasetView(self)

    def __getitem__(self, index: int):
        from .sequence import SingleObjectTrackingDatasetSequenceViewer
        return SingleObjectTrackingDatasetSequenceViewer(self, self.sequences[index])

    def __len__(self):
        return len(self.sequences)

    def getConstructor(self):
        from Dataset.SOT.Base.constructor import SingleObjectTrackingDatasetConstructor
        return SingleObjectTrackingDatasetConstructor(self)

    def hasAttributeCategory(self):
        return self.attributes['has_object_category_attr']

    def hasAttibuteFPS(self):
        return self.attributes['has_fps_attr']

    def getAttribute(self, name):
        return self.attributes[name]

    def hasAttribute(self, name):
        return name in self.attributes

    def getAttributes(self):
        return self.attributes

    def getSequenceIndexByName(self, name: str):
        indices = []
        for index, sequence in enumerate(self.sequences):
            if sequence.name == name:
                indices.append(index)
        return indices

    def getModifier(self):
        from .modifier import SingleObjectDatasetModifier
        return SingleObjectDatasetModifier(self)
