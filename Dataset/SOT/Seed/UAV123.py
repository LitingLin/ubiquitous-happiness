from Dataset.Base.factory_seed import BaseSeed
from Dataset.Type.data_split import DataSplit


class UAV123_Seed(BaseSeed):
    def __init__(self, root_path: str=None):
        if root_path is None:
            root_path = self.get_path_from_config('UAV123_PATH')
        super(UAV123_Seed, self).__init__('UAV123', root_path, DataSplit.Full, 1)

    def construct(self, constructor):
        from .Impl.UAV123 import construct_UAV123
        construct_UAV123(constructor, self)
