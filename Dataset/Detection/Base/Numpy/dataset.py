from typing import List, Dict
import numpy as np
import pathlib
from Dataset.DataSplit import DataSplit
from Utils.deprecator import deprecated


@deprecated
class DetectionDataset_Numpy:
    name: str
    root_path: pathlib.Path

    image_paths: List[pathlib.Path]
    bounding_boxes: np.ndarray
    category_ids: np.ndarray

    image_attributes_indices: np.ndarray

    category_names: List
    category_name_id_mapper: Dict

    data_type: DataSplit
    structure_version: int
    data_version: int

    def __init__(self):
        self.structure_version = 1
        self.data_type = DataSplit.Full

    def setRootPath(self, root_path: str):
        self.root_path = pathlib.Path(root_path)

    def getName(self):
        return self.name

    def getRootPath(self):
        return str(self.root_path)

    def __len__(self):
        return self.getNumberOfImages()

    def __getitem__(self, index: int):
        return self.getImage(index)

    def getNumberOfImages(self):
        return self.image_attributes_indices.shape[0]

    def getImage(self, index: int):
        from Dataset.Detection.Base.Numpy.image import DetectionDatasetImageViewer_Numpy
        attribute_index = self.image_attributes_indices[index]
        if index == self.getNumberOfImages() - 1:
            length = self.bounding_boxes.shape[0] - attribute_index
        else:
            length = self.image_attributes_indices[index + 1] - attribute_index
        return DetectionDatasetImageViewer_Numpy(self, index, attribute_index, length)

    def getCategoryNameList(self):
        return self.category_names

    def getCategoryName(self, id_: int):
        return self.category_names[id_]

    def getCategoryId(self, name: str):
        return self.category_name_id_mapper[name]

    def getConstructor(self):
        from Dataset.Detection.Base.Numpy.constructor import DetectionDatasetConstructor_Numpy
        return DetectionDatasetConstructor_Numpy(self)

    def getFlattenView(self):
        from .flatten_view import DetectionDataset_Numpy_FlattenView
        return DetectionDataset_Numpy_FlattenView(self)
