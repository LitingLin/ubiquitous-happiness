from typing import Dict, List, Any
from Dataset.DataSplit import DataSplit


class MultipleObjectTrackingDataset:
    root_path: str
    name: str
    attributes: Dict[str, Any]
    sequences: List['MultipleObjectTrackingDatasetSequence']

    category_name_id_mapper: Dict[str, int]
    category_names: List[str]

    def __init__(self):
        self.data_version = None
        self.structure_version = 6
        self.root_path = None
        self.filters = []
        self.data_split = DataSplit.Full
        self.name = None
        self.attributes = {}
        self.sequences = []
        self.category_name_id_mapper = {}
        self.category_names = []

    def getRootPath(self):
        return str(self.root_path)

    def setRootPath(self, root_path: str):
        import pathlib
        root_path = pathlib.Path(root_path)
        self.root_path = str(root_path)

    def getCategoryIdByName(self, name: str):
        return self.category_name_id_mapper[name]

    def getCategoryNameList(self):
        return self.category_names

    def getName(self):
        return self.name

    def getAttribute(self, name):
        return self.attributes[name]

    def hasAttribute(self, name):
        return name in self.attributes

    def getAttributes(self):
        return self.attributes

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index: int):
        from .sequence import MultipleObjectTrackingDatasetSequenceView
        return MultipleObjectTrackingDatasetSequenceView(self, self.sequences[index])

    def getConstructor(self):
        from .constructor import MultipleObjectTrackingDatasetConstructor
        return MultipleObjectTrackingDatasetConstructor(self)

    def hasAttibuteFPS(self):
        return self.attributes['has_fps_attr']

    def getSequenceIndexByName(self, name: str):
        indices = []
        for index, sequence in enumerate(self.sequences):
            if sequence.name == name:
                indices.append(index)
        return indices

    def getModifier(self):
        from .modifier import MultipleObjectTrackingDatasetModifier
        return MultipleObjectTrackingDatasetModifier(self)
