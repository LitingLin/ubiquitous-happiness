from Dataset.MOT.Base.visitor import MultipleObjectTrackingDatasetVisitor
from .sequence import MultipleObjectTrackingSiamDatasetSequenceVisitor


class MultipleObjectTrackingSiamDatasetVisitor(MultipleObjectTrackingDatasetVisitor):
    def __getitem__(self, index):
        return MultipleObjectTrackingSiamDatasetSequenceVisitor(self.dataset, self.dataset.sequences[index])
