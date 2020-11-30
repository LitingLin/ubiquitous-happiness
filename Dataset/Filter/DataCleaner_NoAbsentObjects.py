class DataCleaner_NoAbsentObjects:
    def __str__(self):
        return "DataCleaner_NoAbsentObjects"

    def __eq__(self, other):
        return isinstance(other, DataCleaner_NoAbsentObjects)

    def __repr__(self):
        return str(self)
