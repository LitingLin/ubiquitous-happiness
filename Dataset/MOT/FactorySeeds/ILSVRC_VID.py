from Dataset.DataSplit import DataSplit
from Dataset._base_seed import BaseSeed


class ILSVRC_VID_Seed(BaseSeed):
    def __init__(self, root_path: str=None, data_split: DataSplit = DataSplit.Training | DataSplit.Validation):
        if root_path is None:
            root_path = self._getPathFromConfig('ILSVRC_VID_PATH')
        super(ILSVRC_VID_Seed, self).__init__('ILSVRC_VID', root_path, data_split, 1)

    def construct(self, constructor):
        from ..Sources.ILSVRC_VID import construct_ILSVRC_VID
        construct_ILSVRC_VID(constructor, self)
