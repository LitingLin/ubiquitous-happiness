from Dataset.Detection.Base.Numpy.dataset import DetectionDataset_Numpy


class DetectionDatasetObjectViewer_Numpy:
    def __init__(self, dataset: DetectionDataset_Numpy, index: int, attribute_index: int):
        self.dataset = dataset
        self.index = index
        self.attribute_index = attribute_index

    def getImagePath(self):
        return str(self.dataset.root_path.joinpath(self.dataset.image_paths[self.index]))

    def getBoundingBox(self):
        return self.dataset.bounding_boxes[self.attribute_index, :]

    def getCategoryName(self):
        return self.dataset.category_names[self.getCategoryId()]

    def getCategoryId(self):
        return self.dataset.category_ids[self.attribute_index]

    def __iter__(self):
        yield from [self.getBoundingBox(), self.getCategoryName()]
