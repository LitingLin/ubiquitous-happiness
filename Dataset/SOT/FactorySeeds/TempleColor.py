from Dataset._base_seed import BaseSeed
from Dataset.DataSplit import DataSplit


class TempleColor_Seed(BaseSeed):
    def __init__(self, root_path: str=None):
        if root_path is None:
            root_path = self._getPathFromConfig('TempleColor_PATH')
        super(TempleColor_Seed, self).__init__('TempleColor-128', root_path, DataSplit.Full, 1)

    def construct(self, constructor):
        from ..Sources.TempleColor import construct_TempleColor
        construct_TempleColor(constructor, self)
