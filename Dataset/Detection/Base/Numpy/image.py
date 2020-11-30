from Dataset.Detection.Base.Numpy.dataset import DetectionDataset_Numpy

class DetectionDatasetImageViewer_Numpy:
    dataset: DetectionDataset_Numpy

    def __init__(self, dataset: DetectionDataset_Numpy, index: int, attribute_index: int, length: int):
        self.dataset = dataset
        self.index = index
        self.attribute_index = attribute_index
        self.length = length

    def __len__(self):
        return self.length

    def getImagePath(self):
        return str(self.dataset.root_path.joinpath(self.dataset.image_paths[self.index]))

    def getNumberObjects(self):
        return self.length

    def getObject(self, index: int):
        if index >= self.length:
            raise IndexError
        from Dataset.Detection.Base.Numpy.object import DetectionDatasetObjectViewer_Numpy
        return DetectionDatasetObjectViewer_Numpy(self.dataset, self.index, self.attribute_index + index)

    def __getitem__(self, index: int):
        return self.getObject(index)
