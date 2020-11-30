from Dataset._base_seed import BaseSeed
from Dataset.DataSplit import DataSplit


class VOT2015_Seed(BaseSeed):
    def __init__(self, root_path: str=None):
        if root_path is None:
            root_path = self._getPathFromConfig('VOT2015_PATH')
        super(VOT2015_Seed, self).__init__('VOT2015', root_path, DataSplit.Full, 1)

    def construct(self, constructor):
        from ..Sources.VOT2015 import constructVOT2015
        constructVOT2015(constructor, self)
