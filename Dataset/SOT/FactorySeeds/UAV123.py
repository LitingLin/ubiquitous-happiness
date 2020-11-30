from Dataset._base_seed import BaseSeed
from Dataset.DataSplit import DataSplit


class UAV123_Seed(BaseSeed):
    def __init__(self, root_path: str=None):
        if root_path is None:
            root_path = self._getPathFromConfig('UAV123_PATH')
        super(UAV123_Seed, self).__init__('UAV123', root_path, DataSplit.Full, 2)

    def construct(self, constructor):
        from ..Sources.UAV123 import construct_UAV123
        construct_UAV123(constructor, self)
