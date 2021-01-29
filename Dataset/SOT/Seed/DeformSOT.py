from Dataset.Base.factory_seed import BaseSeed
from Dataset.Type.data_split import DataSplit


class DeformSOT_Seed(BaseSeed):
    def __init__(self, root_path: str=None):
        if root_path is None:
            root_path = self.get_path_from_config('DeformSOT_PATH')
        super(DeformSOT_Seed, self).__init__('Deform-SOT', root_path, DataSplit.Full, 1)

    def construct(self, constructor):
        from .Impl.DeformSOT import construct_DeformSOT
        construct_DeformSOT(constructor, self)
