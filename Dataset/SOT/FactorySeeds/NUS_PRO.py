from Dataset._base_seed import BaseSeed
from Dataset.DataSplit import DataSplit


class NUSPRO_Seed(BaseSeed):
    def __init__(self, root_path: str=None):
        if root_path is None:
            root_path = self._getPathFromConfig('NUSPRO_PATH')
        super(NUSPRO_Seed, self).__init__('NUS-PRO', root_path, DataSplit.Full, 2)

    def construct(self, constructor):
        from ..Sources.NUS_PRO import construct_NUSPRO
        construct_NUSPRO(constructor, self)
