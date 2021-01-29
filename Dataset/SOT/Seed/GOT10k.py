from Dataset.Base.factory_seed import BaseSeed
from Dataset.Type.data_split import DataSplit


class GOT10k_Seed(BaseSeed):
    def __init__(self, root_path: str=None, data_split: DataSplit = DataSplit.Training | DataSplit.Validation):
        if root_path is None:
            root_path = self.get_path_from_config('GOT10k_PATH')
        super(GOT10k_Seed, self).__init__('GOT-10k', root_path, data_split, 1)

    def construct(self, constructor):
        from .Impl.GOT10k import construct_GOT10k
        construct_GOT10k(constructor, self)
