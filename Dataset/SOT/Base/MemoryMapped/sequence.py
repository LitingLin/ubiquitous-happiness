from .dataset import SingleObjectTrackingDataset_MemoryMapped


class SingleObjectTrackingDatasetSequence_MemoryMapped:
    def __init__(self, dataset: SingleObjectTrackingDataset_MemoryMapped, index_of_sequence: int, attribute_index: int, length: int):
        self.dataset = dataset
        self.attributes_index = attribute_index
        self.length = length
        self.index = index_of_sequence

    def getName(self):
        return self.dataset.sequence_names[self.index]

    def getCategoryId(self):
        return self.dataset.sequence_category_ids[self.index]

    def getCategoryName(self):
        return self.dataset.category_names[self.getCategoryId()]

    def getFrame(self, index: int):
        if index >= len(self):
            raise IndexError
        from .frame import SingleObjectTrackingDatasetFrame_MemoryMapped
        return SingleObjectTrackingDatasetFrame_MemoryMapped(self.dataset, self.attributes_index + index)

    def __getitem__(self, index: int):
        return self.getFrame(index)

    def __len__(self):
        return self.length

    def getAllBoundingBox(self):
        return self.dataset.bounding_boxes[self.attributes_index: self.attributes_index + self.length, :]

    def hasAttributeCategory(self):
        return self.dataset.hasAttributeCategory()

    def hasAttributeFPS(self):
        return self.dataset.hasAttibuteFPS()

    def getFPS(self):
        return self.dataset.sequence_fps_s[self.index]
