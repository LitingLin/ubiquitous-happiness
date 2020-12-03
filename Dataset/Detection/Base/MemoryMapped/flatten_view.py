import numpy as np

class DetectionDataset_MemoryMapped_FlattenView:
    def __init__(self, dataset):
        self.dataset = dataset
        self.flatten_sequence_indices = np.zeros(shape=(dataset.bounding_boxes.matrix.shape[0],), dtype=np.uint32)
        number_of_sequences = len(dataset)
        for index_of_sequence in range(number_of_sequences):
            begin_index = self.dataset.image_attributes_indices[index_of_sequence]
            if index_of_sequence == number_of_sequences - 1:
                end_index = self.dataset.bounding_boxes.matrix.shape[0]
            else:
                end_index = self.dataset.image_attributes_indices[index_of_sequence + 1]
            self.flatten_sequence_indices[begin_index: end_index] = index_of_sequence

    def getName(self):
        return self.dataset.name

    def getRootPath(self):
        return self.dataset.root_path

    def __len__(self):
        return self.getNumberOfImages()

    def __getitem__(self, index: int):
        return self.getImage(index)

    def getNumberOfImages(self):
        return self.flatten_sequence_indices.shape[0]

    def getImage(self, index: int):
        if index >= len(self):
            raise IndexError
        from .object import DetectionDatasetObjectView_MemoryMapped
        return DetectionDatasetObjectView_MemoryMapped(self.dataset, self.flatten_sequence_indices[index], index)

    def getCategoryNameList(self):
        return self.dataset.category_id_name_mapper.values()

    def getCategoryName(self, id_: int):
        return self.dataset.category_id_name_mapper[id_]
