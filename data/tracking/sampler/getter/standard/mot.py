from Dataset.MOT.Storage.MemoryMapped.dataset import MultipleObjectTrackingDatasetSequence_MemoryMapped, MultipleObjectTrackingDatasetFrame_MemoryMapped, MultipleObjectTrackingDatasetSequenceObject_MemoryMapped, MultipleObjectTrackingDatasetFrameObject_MemoryMapped


class MultipleObjectTrackingDatasetStandardDataGetter:
    def __call__(self, sequence: MultipleObjectTrackingDatasetSequence_MemoryMapped, frame: MultipleObjectTrackingDatasetFrame_MemoryMapped, sequence_object: MultipleObjectTrackingDatasetSequenceObject_MemoryMapped, frame_object: MultipleObjectTrackingDatasetFrameObject_MemoryMapped):
        if frame_object is None:
            return frame.get_image_path(), None, False
        else:
            if sequence.has_bounding_box_validity_flag():
                return frame.get_image_path(), frame_object.get_bounding_box(), frame_object.get_bounding_box_validity_flag()
            else:
                return frame.get_image_path(), frame_object.get_bounding_box(), True
