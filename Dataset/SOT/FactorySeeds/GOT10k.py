from Dataset._base_seed import BaseSeed
from Dataset.DataSplit import DataSplit


class GOT10k_Seed(BaseSeed):
    def __init__(self, root_path: str=None, data_split: DataSplit = DataSplit.Training | DataSplit.Validation):
        if root_path is None:
            root_path = self._getPathFromConfig('GOT10k_PATH')
        super(GOT10k_Seed, self).__init__('GOT-10k', root_path, data_split, 2)

    def construct(self, constructor):
        from ..Sources.GOT10k import construct_GOT10k
        construct_GOT10k(constructor, self)
