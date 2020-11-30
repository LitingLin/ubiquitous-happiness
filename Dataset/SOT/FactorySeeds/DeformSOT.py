from Dataset._base_seed import BaseSeed
from Dataset.DataSplit import DataSplit


class DeformSOT_Seed(BaseSeed):
    def __init__(self, root_path: str=None):
        if root_path is None:
            root_path = self._getPathFromConfig('DeformSOT_PATH')
        super(DeformSOT_Seed, self).__init__('Deform-SOT', root_path, DataSplit.Full, 1)

    def construct(self, constructor):
        from ..Sources.DeformSOT import construct_DeformSOT
        construct_DeformSOT(constructor, self)
