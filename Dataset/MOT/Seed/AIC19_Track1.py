from Dataset.Type.data_split import DataSplit
from Dataset.Base.factory_seed import BaseSeed


class AIC19_Track1_Seed(BaseSeed):
    def __init__(self, root_path: str=None):
        if root_path is None:
            root_path = self.get_path_from_config('AIC19_Track1_PATH')
        super(AIC19_Track1_Seed, self).__init__('AIC19_Track1', root_path, DataSplit.Training, 1)

    def construct(self, constructor):
        from .Impl.AIC19_Track1 import construct_AIC19Track1
        construct_AIC19Track1(constructor, self)
