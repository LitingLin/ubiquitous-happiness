from Utils.join_and_get_platform_style_path import join_and_get_platform_style_path


class SingleObjectTrackingDatasetFrame_MemoryMapped:
    def __init__(self, dataset, attribute_index):
        self.dataset = dataset
        self.attribute_index = attribute_index

    def getImagePath(self):
        return join_and_get_platform_style_path(self.dataset.root_path, self.dataset.image_paths[self.attribute_index])

    def getBoundingBox(self):
        return self.dataset.bounding_boxes[self.attribute_index, :]

    def getAttributeIsPresent(self):
        bounding_box = self.getBoundingBox()
        return bounding_box[2] > 0 and bounding_box[3] > 0

    def hasAttributes(self):
        return True

    def __iter__(self):
        yield from [self.getImagePath(), self.getBoundingBox()]
