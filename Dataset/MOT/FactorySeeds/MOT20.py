from Dataset.DataSplit import DataSplit
from Dataset._base_seed import BaseSeed


class MOT20_Seed(BaseSeed):
    def __init__(self, root_path: str=None):
        if root_path is None:
            root_path = self._getPathFromConfig('MOT20_PATH')
        super(MOT20_Seed, self).__init__('MOT20', root_path, DataSplit.Full, 1)

    def construct(self, constructor):
        from ..Sources.MOT20 import construct_MOT20
        construct_MOT20(constructor, self)
