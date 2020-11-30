from Dataset._base_seed import BaseSeed
from Dataset.DataSplit import DataSplit


class DTB70_Seed(BaseSeed):
    def __init__(self, root_path: str=None):
        if root_path is None:
            root_path = self._getPathFromConfig('DTB70_PATH')
        super(DTB70_Seed, self).__init__('DTB70', root_path, DataSplit.Full, 1)

    def construct(self, constructor):
        from ..Sources.DTB70 import construct_DTB70
        construct_DTB70(constructor, self)
