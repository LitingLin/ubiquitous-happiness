from Dataset.MOT.Base.dataset import MultipleObjectTrackingDataset
from Dataset.MOT.Base.sequence import MultipleObjectTrackingDatasetSequence, MultipleObjectTrackingDatasetSequenceView
from Dataset.MOT.Base.object import MultipleObjectTrackingDatasetSequenceObjectView, MultipleObjectTrackingDatasetFrameObjectView


class MultipleObjectTrackingDatasetSequenceTrackView:
    def __init__(self, dataset: MultipleObjectTrackingDataset, sequence: MultipleObjectTrackingDatasetSequence, object_id: int):
        self.dataset = dataset
        self.sequence = sequence
        self.object_id = object_id
        self.object_attributes = self.sequence.object_id_attributes_mapper[object_id]

    def getObjectId(self):
        return self.object_id

    def getSequence(self):
        return MultipleObjectTrackingDatasetSequenceView(self.dataset, self.sequence)

    def getObject(self):
        return MultipleObjectTrackingDatasetSequenceObjectView(self.dataset, self.sequence, self.object_id)

    def getNumberOfFrames(self):
        return len(self.object_attributes.frame_indices)

    def getFrameIndices(self):
        return self.object_attributes.frame_indices

    def __getitem__(self, index: int):
        frame = self.sequence.frames[self.object_attributes.frame_indices[index]]
        frame_object_attributes = frame.objects[self.object_id]
        return MultipleObjectTrackingDatasetFrameObjectView(self.dataset,
                                                            frame,
                                                            self.object_id,
                                                            self.object_attributes,
                                                            frame_object_attributes)

    def __len__(self):
        return self.getNumberOfFrames()


class MultipleObjectTrackingDatasetSequenceTrackIterator:
    def __init__(self, dataset: MultipleObjectTrackingDataset, sequence: MultipleObjectTrackingDatasetSequence):
        self.dataset = dataset
        self.sequence = sequence

    def __len__(self):
        return len(self.sequence.object_ids)

    def __getitem__(self, index: int):
        return MultipleObjectTrackingDatasetSequenceTrackView(self.dataset, self.sequence, self.sequence.object_ids[index])
