from typing import List, Dict
from Dataset.DataSplit import DataSplit


class SingleObjectTrackingDataset:
    sequences: List['SingleObjectTrackingDatasetSequence']
    root_path: str
    category_id_name_mapper: Dict[int, str]
    name: str
    attributes: Dict

    def __init__(self):
        self.data_version = None
        self.structure_version = 6
        self.data_split = DataSplit.Full
        self.filters = []
        self.sequences = []
        self.root_path = None
        self.category_id_name_mapper = {}
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
        return self.category_id_name_mapper.values()

    def getNumberOfCategories(self):
        return len(self.category_id_name_mapper)

    def getMaxCategoryId(self):
        return max(self.category_id_name_mapper.keys())

    def getCategoryName(self, id_: int):
        return self.category_id_name_mapper[id_]

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
