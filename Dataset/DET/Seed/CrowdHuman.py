from Dataset.Base.factory_seed import BaseSeed
from Dataset.Type.data_split import DataSplit


class CrowdHuman_Seed(BaseSeed):
    def __init__(self, root_path: str=None, data_split: DataSplit=DataSplit.Training | DataSplit.Validation, fine_anno_only=True):
        assert fine_anno_only
        name = 'CrowdHuman'
        if fine_anno_only:
            name += '_fine_anno_only'
        if root_path is None:
            root_path = self.get_path_from_config('CrowdHuman_PATH')
        super(CrowdHuman_Seed, self).__init__('CrowdHuman', root_path, data_split, 1)

    def construct(self, constructor):
        from .Impl.CrowdHuman import construct_CrowdHuman
        construct_CrowdHuman(constructor, self)
