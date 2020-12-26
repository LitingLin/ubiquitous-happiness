from Dataset.Detection.Base.MemoryMapped.dataset import DetectionDataset_MemoryMapped
from Utils.join_and_get_platform_style_path import join_and_get_platform_style_path


class DetectionDatasetImageView_MemoryMapped:
    dataset: DetectionDataset_MemoryMapped

    def __init__(self, dataset: DetectionDataset_MemoryMapped, index: int, attribute_index: int, length: int):
        self.dataset = dataset
        self.index = index
        self.attribute_index = attribute_index
        self.length = length

    def __len__(self):
        return self.length

    def getImagePath(self):
        return join_and_get_platform_style_path(self.dataset.root_path, self.dataset.image_paths[self.index])

    def getNumberObjects(self):
        return self.length

    def getObject(self, index: int):
        if index >= self.length:
            raise IndexError
        from Dataset.Detection.Base.MemoryMapped.object import DetectionDatasetObjectView_MemoryMapped
        return DetectionDatasetObjectView_MemoryMapped(self.dataset, self.index, self.attribute_index + index)

    def getAllCategoryId(self):
        return self.dataset.category_ids[self.attribute_index: self.attribute_index + self.length]

    def getAllBoundingBox(self):
        return self.dataset.bounding_boxes[self.attribute_index: self.attribute_index + self.length, :]

    def getAllAttributeIsPresent(self):
        return self.dataset.is_presents[self.attribute_index: self.attribute_index + self.length]

    def getImageSize(self):
        return self.dataset.image_sizes[self.index, :]

    def __getitem__(self, index: int):
        return self.getObject(index)

    def hasAttributeCategory(self):
        return self.dataset.hasAttributeCategory()

    def hasAttributeIsPresent(self):
        return self.dataset.hasAttibuteIsPresent()
