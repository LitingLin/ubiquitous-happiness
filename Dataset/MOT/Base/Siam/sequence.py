from Dataset.MOT.Base.sequence import MultipleObjectTrackingDatasetSequenceVisitor, MultipleObjectTrackingDatasetSequence
from typing import List, Dict


class MultipleObjectTrackingSiamDatasetSequence(MultipleObjectTrackingDatasetSequence):
    cropped_images: Dict[int, List]

    def __init__(self):
        super(MultipleObjectTrackingSiamDatasetSequence, self).__init__()
        self.cropped_images = {}


class MultipleObjectTrackingSiamDatasetSequenceVisitor(MultipleObjectTrackingDatasetSequenceVisitor):
    def __init__(self, dataset: 'MultipleObjectTrackingSiamDataset', sequence: MultipleObjectTrackingSiamDatasetSequence):
        self.dataset = dataset
        self.sequence = sequence

    def getCroppedImage(self, exemplar_size: int, frame_index: int):
        return self.sequence.cropped_images[exemplar_size][frame_index]
