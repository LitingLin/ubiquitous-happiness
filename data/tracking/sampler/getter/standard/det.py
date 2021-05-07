from Dataset.DET.Storage.MemoryMapped.dataset import DetectionDatasetImage_MemoryMapped, DetectionDatasetObject_MemoryMapped


def detection_dataset_standard_data_getter(image: DetectionDatasetImage_MemoryMapped, object_: DetectionDatasetObject_MemoryMapped):
    return image.get_image_path(), image.get_image_size(), object_.get_bounding_box(), object_.get_bounding_box_validity_flag() if image.has_bounding_box_validity_flag() else True
