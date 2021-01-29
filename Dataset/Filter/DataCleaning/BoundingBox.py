from Dataset.Filter._common import _BaseFilter


class DataCleaning_BoundingBox(_BaseFilter):
    def __init__(self, fit_in_image_size:bool=False, format:str=None, update_validity:bool=False, remove_non_validity_objects=False, remove_empty_annotation_objects=False):
        self.fit_in_image_size = fit_in_image_size
        self.format = format
        self.update_validity = update_validity
        self.remove_non_validity_objects = remove_non_validity_objects
        self.remove_empty_annotation_objects = remove_empty_annotation_objects
