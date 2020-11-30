from .dataset import SingleObjectTrackingDataset_MemoryMapped
from typing import List
import numpy as np


class SingleObjectTrackingDatasetView_MemoryMapped:
    sequence_indices: np.ndarray

    def __init__(self, dataset: SingleObjectTrackingDataset_MemoryMapped):
        self.dataset = dataset
        self.clearFilter()

    def clearFilter(self):
        self.sequence_indices = np.arange(len(self.dataset))

    def applyIndicesFilter(self, indices: List[int]):
        indices = np.array(indices)
        index_selector = np.zeros(self.sequence_indices.shape, np.bool)
        index_selector[indices] = True
        self.sequence_indices = self.sequence_indices[index_selector]

    def getName(self):
        return self.dataset.name

    def getCategoryNameList(self):
        return self.dataset.category_names

    def getNumberOfCategories(self):
        return len(self.dataset.category_names)

    def getCategoryName(self, id_: int):
        return self.dataset.category_names[id_]

    def __len__(self):
        return self.sequence_indices.shape[0]

    def __getitem__(self, index: int):
        from .sequence import SingleObjectTrackingDatasetSequence_MemoryMapped

        index = self.sequence_indices[index]
        attributes_index = self.dataset.sequence_attributes_indices[index * 2]
        length = self.dataset.sequence_attributes_indices[index * 2 + 1] - attributes_index
        return SingleObjectTrackingDatasetSequence_MemoryMapped(self.dataset, index, attributes_index, length)

    def hasAttributeCategory(self):
        return self.dataset.hasAttributeCategory()

    def hasAttibuteFPS(self):
        return self.dataset.hasAttibuteFPS()
