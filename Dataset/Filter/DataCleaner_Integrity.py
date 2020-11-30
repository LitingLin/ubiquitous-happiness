class DataCleaner_Integrity:
    def __init__(self, no_zero_annotations=True, no_zero_size_image=True):
        self.no_zero_annotations = no_zero_annotations
        self.no_zero_size_image = no_zero_size_image

    def __str__(self):
        return "DataCleaner_BoundingBox-{}.{}".format(self.no_zero_annotations, self.no_zero_size_image)

    def __eq__(self, other):
        if not isinstance(other, DataCleaner_Integrity):
            return False
        return self.no_zero_annotations == other.no_zero_annotations and self.no_zero_size_image == other.no_zero_size_image

    def __repr__(self):
        return str(self)
