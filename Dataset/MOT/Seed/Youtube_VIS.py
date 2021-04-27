from Dataset.Base.factory_seed import BaseSeed
from Dataset.Type.data_split import DataSplit


class Youtube_VIS_Seed(BaseSeed):
    def __init__(self, root_path: str=None, data_split: DataSplit=DataSplit.Training):
        if root_path is None:
            root_path = self.get_path_from_config('Youtube_VIS_PATH')
        super(Youtube_VIS_Seed, self).__init__('Youtube-VIS', root_path, data_split, 1)

    def construct(self, constructor):
        from Dataset.MOT.Seed.Impl.YouTube_VIS import construct_Youtube_VIS
        construct_Youtube_VIS(constructor, self)
