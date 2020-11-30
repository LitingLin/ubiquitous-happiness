from typing import List
from .Base.dataset import DetectionDataset
import numpy as np


class MultiDatasetView:
    def __init__(self, name: str, datasets: List[DetectionDataset], category_names: List[str]=None):
        self.has_attribute_is_present = datasets[0].hasAttibuteIsPresent()
        self.has_attribute_category = datasets[0].hasAttributeCategory()
        for dataset in datasets:
            assert self.has_attribute_is_present == dataset.hasAttibuteIsPresent()
            assert self.has_attribute_category == dataset.hasAttributeCategory()

        self.indirect_indices = None

        self.datasets = datasets
        self.name = name
        self.indices = []
        length = 0
        for dataset in datasets:
            length += len(dataset)
            self.indices.append(length)
        self.indices = np.array(self.indices)
        self.length = length
        if category_names is not None:
            self.category_names = category_names
            self.category_name_id_mapper = {name: id_ for id_, name in enumerate(self.category_names)}
        elif self.has_attribute_category:
            category_names = set()
            for dataset in datasets:
                dataset_category_names = dataset.getCategoryNameList()
                for dataset_category_name in dataset_category_names:
                    category_names.add(dataset_category_name)
            self.category_names = list(category_names)
            self.category_name_id_mapper = {name: id_ for id_, name in enumerate(self.category_names)}

    def getCategoryName(self, id_: int):
        return self.category_names[id_]

    def getCategoryId(self, name: str):
        return self.category_name_id_mapper[name]

    def getCategoryNameList(self):
        return self.category_names

    def getNumberOfCategories(self):
        return len(self.category_names)

    def __len__(self):
        return self.length

    def getImage(self, index: int):
        return self[index]

    def __getitem__(self, index: int):
        if index >= len(self):
            raise IndexError
        if self.indirect_indices is not None:
            index = self.indirect_indices[index]
        dataset_index = np.searchsorted(self.indices, index, side='right')
        if dataset_index == 0:
            image_index = index
        else:
            image_index = index - self.indices[dataset_index - 1]
        return self.datasets[dataset_index][image_index]

    def getNumberOfImages(self):
        return len(self)

    def getName(self):
        return self.name

    def hasAttributeCategory(self):
        return self.has_attribute_category

    def hasAttibuteIsPresent(self):
        return self.has_attribute_is_present

    def sortByImageRatio(self):
        assert self.indirect_indices is None
        image_sizes = self.getImageShapes()
        ratio = image_sizes[:, 1] / image_sizes[:, 0]
        indices = ratio.argsort()
        self.indirect_indices = indices

    def getImageShapes(self):
        image_sizes = []
        for image in self:
            image_sizes.append(image.getImageSize())
        image_sizes = np.array(image_sizes)
        if self.indirect_indices is not None:
            return image_sizes[self.indirect_indices]
        else:
            return image_sizes
