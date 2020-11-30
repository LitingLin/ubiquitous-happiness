from typing import List, Dict
import numpy as np
from Dataset.DataSplit import DataSplit
from Dataset.Types.MemoryMapped.digit_array import DigitMatrixMemoryMapped
from Dataset.Types.MemoryMapped.string_array import StringArrayMemoryMapped
import os
from Dataset.CacheService.common import _getCachePath


class DetectionDataset_MemoryMapped:
    name: str
    root_path: str
    image_paths: StringArrayMemoryMapped
    bounding_boxes: DigitMatrixMemoryMapped

    image_attributes_indices: np.ndarray
    image_sizes: np.ndarray

    data_type: DataSplit
    structure_version: int
    data_version: int

    # the following is optional
    category_ids: np.ndarray
    category_names: List
    category_name_id_mapper: Dict

    is_presents: np.ndarray

    def __init__(self):
        self.structure_version = 3
        self.filters = []
        self.data_type = DataSplit.Full
        self.attributes = {}

    def setRootPath(self, root_path: str):
        self.root_path = root_path

    def getName(self):
        return self.name

    def getRootPath(self):
        return self.root_path

    def __len__(self):
        return self.getNumberOfImages()

    def __getitem__(self, index: int):
        return self.getImage(index)

    def getNumberOfImages(self):
        return self.image_attributes_indices.shape[0]

    def __getstate__(self):
        state = {}
        for key, value in self.__dict__.items():
            if key == 'image_paths' or key == 'bounding_boxes':
                continue
            state[key] = value
        return state, self.image_paths.indices, self.bounding_boxes.matrix.dtype, self.bounding_boxes.matrix.shape

    def __setstate__(self, state):
        self.__dict__.update(state[0])
        cache_path, cache_file_prefix = _getCachePath(self)
        image_paths_memory_mapped_file_name = os.path.join(cache_path, cache_file_prefix + '-image_paths.numpy')
        bounding_boxes_memory_mapped_file_name = os.path.join(cache_path, cache_file_prefix + '-bounding_boxes.numpy')
        self.image_paths = StringArrayMemoryMapped(image_paths_memory_mapped_file_name, state[1])
        self.bounding_boxes = DigitMatrixMemoryMapped(bounding_boxes_memory_mapped_file_name, state[2], state[3])

    def getImage(self, index: int):
        from Dataset.Detection.Base.MemoryMapped.image import DetectionDatasetImageView_MemoryMapped
        attribute_index = self.image_attributes_indices[index]
        if index == self.getNumberOfImages() - 1:
            length = self.bounding_boxes.matrix.shape[0] - attribute_index
        else:
            length = self.image_attributes_indices[index + 1] - attribute_index
        return DetectionDatasetImageView_MemoryMapped(self, index, attribute_index, length)

    def getCategoryNameList(self):
        return self.category_names

    def getCategoryName(self, id_: int):
        return self.category_names[id_]

    def getAttribute(self, name):
        return self.attributes[name]

    def hasAttribute(self, name):
        return name in self.attributes

    def getAttributes(self):
        return self.attributes

    def getCategoryId(self, name: str):
        return self.category_name_id_mapper[name]

    def getConstructor(self):
        from Dataset.Detection.Base.MemoryMapped.constructor import DetectionDatasetConstructor_MemoryMapped
        return DetectionDatasetConstructor_MemoryMapped(self)

    def getFlattenView(self):
        from .flatten_view import DetectionDataset_MemoryMapped_FlattenView
        return DetectionDataset_MemoryMapped_FlattenView(self)

    def hasAttributeCategory(self):
        return hasattr(self, 'category_ids')

    def hasAttibuteIsPresent(self):
        return hasattr(self, 'is_presents')
