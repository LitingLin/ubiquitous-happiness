from Dataset._base_seed import BaseSeed
from Dataset.DataSplit import DataSplit


class ILSVRC_DET_Seed(BaseSeed):
    def __init__(self, root_path: str=None, data_split=DataSplit.Training | DataSplit.Validation):
        if root_path is None:
            root_path = self._getPathFromConfig('ILSVRC_DET_PATH')
        super(ILSVRC_DET_Seed, self).__init__('ILSVRC_DET', root_path, data_split, 2)

    def construct(self, constructor):
        from ..Sources.ILSVRC_DET import construct_ILSVRC_DET
        construct_ILSVRC_DET(constructor, self)
