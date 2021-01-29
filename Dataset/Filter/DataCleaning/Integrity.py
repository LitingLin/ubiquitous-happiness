from Dataset.Filter._common import _BaseFilter


class DataCleaning_Integrity(_BaseFilter):
    def __init__(self, remove_zero_annotation=True, remove_invalid_image=True):
        self.remove_zero_annotation = remove_zero_annotation
        self.remove_invalid_image = remove_invalid_image
