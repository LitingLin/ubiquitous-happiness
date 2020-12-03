from ._common import _BaseFilter


class DataCleaner_Integrity(_BaseFilter):
    def __init__(self, no_zero_annotations=True, no_zero_size_image=True):
        self.no_zero_annotations = no_zero_annotations
        self.no_zero_size_image = no_zero_size_image
