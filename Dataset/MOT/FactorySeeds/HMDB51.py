from Dataset.DataSplit import DataSplit
from Dataset._base_seed import BaseSeed


class HMDB51_Seed(BaseSeed):
    def __init__(self, root_path: str=None, annotation_path: str=None):
        if root_path is None:
            root_path = self._getPathFromConfig('HMDB51_PATH')
        if annotation_path is None:
            annotation_path = self._getPathFromConfig('HMDB51_ANNOTATION_PATH')
        super(HMDB51_Seed, self).__init__('HMDB51', root_path, DataSplit.Full, 3)
        self.annotation_path = annotation_path

    def construct(self, constructor):
        from ..Sources.HMDB51 import construct_HMDB51
        construct_HMDB51(constructor, self)
