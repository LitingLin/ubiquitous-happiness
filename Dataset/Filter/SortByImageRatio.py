class SortByImageRatio:
    def __str__(self):
        return "SortByImageRatio"

    def __eq__(self, other):
        return isinstance(other, SortByImageRatio)

    def __repr__(self):
        return str(self)
