import numpy as np
from typing import List, Dict
from Dataset.Types.MemoryMapped.string_array import StringArrayMemoryMapped
from Dataset.Types.MemoryMapped.digit_array import DigitMatrixMemoryMapped
from Dataset.DataSplit import DataSplit
from Dataset.CacheService.common import _getCachePath
import os


class SingleObjectTrackingDataset_MemoryMapped:
    name: str
    root_path: str

    data_type: DataSplit
    structure_version: int
    data_version: int

    image_paths: StringArrayMemoryMapped
    bounding_boxes: DigitMatrixMemoryMapped
    sequence_names: List[str]

    sequence_attributes_indices: np.ndarray
    # following are optional:
    sequence_category_ids: np.ndarray

    sequence_fps_s: np.ndarray

    category_id_name_mapper: Dict[int, str]

    def __init__(self):
        self.structure_version = 2
        self.filters = []
        self.attributes = {}

    def getConstructor(self):
        from .constructor import SingleObjectTrackingDatasetConstructor_MemoryMapped
        return SingleObjectTrackingDatasetConstructor_MemoryMapped(self)

    def setRootPath(self, path: str):
        self.root_path = path

    def getName(self):
        return self.name

    def hasAttributeCategory(self):
        return hasattr(self, 'sequence_category_ids')

    def hasAttibuteFPS(self):
        return hasattr(self, 'sequence_fps_s')

    def getCategoryNameList(self):
        return self.category_id_name_mapper.values()

    def getNumberOfCategories(self):
        return len(self.category_id_name_mapper)

    def getMaxCategoryId(self):
        return max(self.category_id_name_mapper.keys())

    def getCategoryName(self, id_: int):
        return self.category_id_name_mapper[id_]

    def getAttribute(self, name):
        return self.attributes[name]

    def hasAttribute(self, name):
        return name in self.attributes

    def getAttributes(self):
        return self.attributes

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

    def __len__(self):
        return self.sequence_attributes_indices.shape[0] // 2

    def __getitem__(self, index: int):
        if index >= len(self):
            raise IndexError
        from .sequence import SingleObjectTrackingDatasetSequence_MemoryMapped

        attributes_index = self.sequence_attributes_indices[index * 2]
        length = self.sequence_attributes_indices[index * 2 + 1] - attributes_index
        return SingleObjectTrackingDatasetSequence_MemoryMapped(self, index, attributes_index, length)

    def getView(self):
        from .view import SingleObjectTrackingDatasetView_MemoryMapped
        return SingleObjectTrackingDatasetView_MemoryMapped(self)
