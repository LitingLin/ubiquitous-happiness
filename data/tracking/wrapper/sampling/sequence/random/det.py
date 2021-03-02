class DetectionDatasetSequenceSampler:
    def __call__(self, image, object_):
        return image.get_image_path(), object_.get_bounding_box(), object_.get_bounding_box_validity_flag() if object_.has_bounding_box_validity_flag() else True
