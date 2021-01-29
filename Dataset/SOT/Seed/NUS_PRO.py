from Dataset.Base.factory_seed import BaseSeed
from Dataset.Type.data_split import DataSplit


class NUSPRO_Seed(BaseSeed):
    def __init__(self, root_path: str=None):
        if root_path is None:
            root_path = self.get_path_from_config('NUSPRO_PATH')
        super(NUSPRO_Seed, self).__init__('NUS-PRO', root_path, DataSplit.Full, 2)

    def construct(self, constructor):
        from .Impl.NUS_PRO import construct_NUSPRO
        construct_NUSPRO(constructor, self)
