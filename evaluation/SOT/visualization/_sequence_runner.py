from Dataset.SOT.Storage.MemoryMapped.dataset import SingleObjectTrackingDatasetSequence_MemoryMapped
from evaluation.SOT.util.simple_sequence_prefetcher import get_simple_sequence_data_prefetcher


class SequenceRunner:
    def __init__(self, tracker, sequence: SingleObjectTrackingDatasetSequence_MemoryMapped, image_transform=None):
        self.tracker = tracker
        self.sequence = sequence
        self.image_transform = image_transform
        assert len(sequence) > 1

    def __iter__(self):
        self.index = 0
        self.sequence_data_iter = iter(get_simple_sequence_data_prefetcher(self.sequence))
        return self

    def __next__(self):
        index = self.index
        image, groundtruth_bounding_box, groundtruth_bounding_box_validity_flag = next(self.sequence_data_iter)
        if groundtruth_bounding_box_validity_flag is not None:
            groundtruth_bounding_box_validity_flag = bool(groundtruth_bounding_box_validity_flag)
        if index == 0:
            self.tracker.initialize(image, groundtruth_bounding_box)
            predicted_bounding_box = groundtruth_bounding_box.tolist()
        else:
            predicted_bounding_box, _ = self.tracker.track(image)
        self.index += 1
        image = image.permute(1, 2, 0)
        if self.image_transform is not None:
            image = self.image_transform(image)

        return (index, image, groundtruth_bounding_box.tolist(), groundtruth_bounding_box_validity_flag, predicted_bounding_box)
