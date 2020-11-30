import numpy as np

class DetectionDataset_Numpy_FlattenView:
    def __init__(self, dataset):
        self.dataset = dataset

    def getName(self):
        return self.dataset.name

    def getRootPath(self):
        return str(self.dataset.root_path)

    def __len__(self):
        return self.getNumberOfImages()

    def __getitem__(self, index: int):
        return self.getImage(index)

    def getNumberOfImages(self):
        return self.dataset.bounding_boxes.shape[0]

    def getImage(self, index: int):
        if index >= len(self):
            raise IndexError
        sequence_index = np.searchsorted(self.dataset.image_attributes_indices, index, side='right')
        if sequence_index == 0:
            raise IndexError
        sequence_index -= 1
        from .object import DetectionDatasetObjectViewer_Numpy
        return DetectionDatasetObjectViewer_Numpy(self.dataset, sequence_index, index)

    def getCategoryNameList(self):
        return self.dataset.category_names

    def getCategoryName(self, id_: int):
        return self.dataset.category_names[id_]

    def getCategoryId(self, name: str):
        return self.dataset.category_name_id_mapper[name]
