from Dataset._base_seed import BaseSeed
from Dataset.DataSplit import DataSplit


class PTB_Seed(BaseSeed):
    def __init__(self, root_path: str=None):
        if root_path is None:
            root_path = self._getPathFromConfig('PTB_PATH')
        super(PTB_Seed, self).__init__('PTB', root_path, DataSplit.Full, 3)

    def construct(self, constructor):
        from ..Sources.PTB import construct_PTB
        construct_PTB(constructor, self)
