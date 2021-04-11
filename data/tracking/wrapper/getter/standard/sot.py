from Dataset.SOT.Storage.MemoryMapped.dataset import SingleObjectTrackingDatasetSequence_MemoryMapped, SingleObjectTrackingDatasetFrame_MemoryMapped

class SingleObjectTrackingDatasetStandardDataGetter:
    def __call__(self, sequence: SingleObjectTrackingDatasetSequence_MemoryMapped, frame: SingleObjectTrackingDatasetFrame_MemoryMapped):
        if sequence.has_bounding_box_validity_flag():
            return frame.get_image_path(), frame.get_bounding_box(), frame.get_bounding_box_validity_flag()
        else:
            return frame.get_image_path(), frame.get_bounding_box(), True
