class _BaseFilter:
    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return type(self) == type(other) and self.__dict__ == other.__dict__

    def __str__(self):
        return f'{self.__class__.__name__}{self.__dict__}'
