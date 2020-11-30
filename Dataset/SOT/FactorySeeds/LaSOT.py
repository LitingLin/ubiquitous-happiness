from Dataset._base_seed import BaseSeed
from Dataset.DataSplit import DataSplit


class LaSOT_Seed(BaseSeed):
    def __init__(self, root_path: str=None, data_split: DataSplit = DataSplit.Training | DataSplit.Validation):
        if root_path is None:
            root_path = self._getPathFromConfig('LaSOT_PATH')
        super(LaSOT_Seed, self).__init__('LaSOT', root_path, data_split, 2)

    def construct(self, constructor):
        from ..Sources.LaSOT import construct_LaSOT
        construct_LaSOT(constructor, self)
