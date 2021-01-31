from Dataset.Type.data_split import DataSplit
from Dataset.Base.factory_seed import BaseSeed


class YoutubeBB_Seed(BaseSeed):
    def __init__(self, root_path: str=None, data_split: DataSplit = DataSplit.Training | DataSplit.Validation, train_csv=None, validation_csv=None):
        if root_path is None:
            root_path = self.get_path_from_config('Youtube_BB_PATH')
        super(YoutubeBB_Seed, self).__init__('Youtube-BB', root_path, data_split, 2)
        self.train_csv = train_csv
        self.validation_csv = validation_csv

    def construct(self, constructor):
        from .Impl.YoutubeBB import construct_YoutubeBB
        construct_YoutubeBB(constructor, self)
