from Dataset.Base.factory_seed import BaseSeed
from Dataset.Type.data_split import DataSplit


class OTB100_Seed(BaseSeed):
    def __init__(self, root_path: str=None):
        if root_path is None:
            root_path = self.get_path_from_config('OTB100_PATH')
        super(OTB100_Seed, self).__init__('OTB100', root_path, DataSplit.Full, 3)

    def construct(self, constructor):
        from .Impl.OTB100 import construct_OTB100
        construct_OTB100(constructor, self, 'otb100')
