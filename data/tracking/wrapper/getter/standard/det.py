from Dataset.DET.Storage.MemoryMapped.dataset import DetectionDatasetImage_MemoryMapped, DetectionDatasetObject_MemoryMapped


class DetectionDatasetStandardDataGetter:
    def __call__(self, image: DetectionDatasetImage_MemoryMapped, object_: DetectionDatasetObject_MemoryMapped):
        return image.get_image_path(), object_.get_bounding_box(), object_.get_bounding_box_validity_flag() if image.has_bounding_box_validity_flag() else True
