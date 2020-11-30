from Dataset.Detection.Base.MemoryMapped.dataset import DetectionDataset_MemoryMapped
from Utils.join_and_get_platform_style_path import join_and_get_platform_style_path


class DetectionDatasetObjectView_MemoryMapped:
    def __init__(self, dataset: DetectionDataset_MemoryMapped, index: int, attribute_index: int):
        self.dataset = dataset
        self.index = index
        self.attribute_index = attribute_index

    def getImagePath(self):
        return join_and_get_platform_style_path(self.dataset.root_path, self.dataset.image_paths[self.index])

    def getBoundingBox(self):
        return self.dataset.bounding_boxes[self.attribute_index, :]

    def getCategoryName(self):
        return self.dataset.category_names[self.getCategoryId()]

    def getCategoryId(self):
        return self.dataset.category_ids[self.attribute_index]

    def hasAttributeIsPresent(self):
        return self.dataset.hasAttibuteIsPresent()

    def getAttributeIsPresent(self):
        return self.dataset.is_presents[self.attribute_index]

    def hasAttributeCategory(self):
        return self.dataset.hasAttributeCategory()
