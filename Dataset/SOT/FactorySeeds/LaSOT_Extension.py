from Dataset._base_seed import BaseSeed
from Dataset.DataSplit import DataSplit


class LaSOT_Extension_Seed(BaseSeed):
    def __init__(self, root_path: str=None):
        if root_path is None:
            root_path = self._getPathFromConfig('LaSOT_Extension_PATH')
        super(LaSOT_Extension_Seed, self).__init__('LaSOT_Extension', root_path, DataSplit.Full, 1)

    def construct(self, constructor):
        from ..Sources.LaSOT_Extension import construct_LaSOT_Extension
        construct_LaSOT_Extension(constructor, self)
