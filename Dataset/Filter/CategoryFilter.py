from typing import List, Dict


class ObjectCategoryFilter:
    r"""
        TODO: unuseful -> useless
            what is useless?
    """
    def __init__(self, includes: List[str]=None, excludes: List[str]=None, mapping: Dict[str, str]=None, remove_unuseful_categories=True):
        self.includes = includes
        self.excludes = excludes
        self.mapping = mapping
        self.remove_unuseful_categories = remove_unuseful_categories

    def __str__(self):
        return 'includes:' + str(self.includes) + 'excludes:' + str(self.excludes) + 'mapping:' + str(self.mapping) + 'remove_unuseful_categories:' + str(self.remove_unuseful_categories)

    def __eq__(self, other):
        if not isinstance(other, ObjectCategoryFilter):
            return False

        return self.includes == other.includes and self.excludes == other.excludes and self.mapping == other.mapping and self.remove_unuseful_categories == other.remove_unuseful_categories

    def __repr__(self):
        return str(self)
