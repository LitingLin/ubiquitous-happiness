from typing import Dict


class CategoryMapping:
    def __init__(self, map_: Dict[str, str]):
        self.map_ = map_

    def __str__(self):
        return 'CategoryMapping: ' + str(self.map_)

    def __eq__(self, other):
        return isinstance(other, CategoryMapping)

    def __repr__(self):
        return str(self)
