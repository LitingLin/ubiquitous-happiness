from Dataset.MOT.Storage.MemoryMapped.dataset import MultipleObjectTrackingDatasetSequence_MemoryMapped, MultipleObjectTrackingDatasetFrame_MemoryMapped, MultipleObjectTrackingDatasetSequenceObject_MemoryMapped, MultipleObjectTrackingDatasetFrameObject_MemoryMapped


def multiple_object_tracking_dataset_standard_data_getter(sequence: MultipleObjectTrackingDatasetSequence_MemoryMapped, frame: MultipleObjectTrackingDatasetFrame_MemoryMapped, sequence_object: MultipleObjectTrackingDatasetSequenceObject_MemoryMapped, frame_object: MultipleObjectTrackingDatasetFrameObject_MemoryMapped):
    if frame_object is None:
        return frame.get_image_path(), frame.get_image_size(), None, False
    else:
        if sequence.has_bounding_box_validity_flag():
            return frame.get_image_path(), frame.get_image_size(), frame_object.get_bounding_box(), frame_object.get_bounding_box_validity_flag()
        else:
            return frame.get_image_path(), frame.get_image_size(), frame_object.get_bounding_box(), True
