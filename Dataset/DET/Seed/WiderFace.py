from Dataset.Base.factory_seed import BaseSeed
from Dataset.Type.data_split import DataSplit


class WiderFace_Seed(BaseSeed):
    def __init__(self, root_path=None, data_split=DataSplit.Training | DataSplit.Validation):
        if root_path is None:
            root_path = self.get_path_from_config('WiderFace_PATH')
        super(WiderFace_Seed, self).__init__('WiderFace', root_path, data_split, 1)

    def construct(self, constructor):
        from .Impl.WiderFace import construct_WiderFace
        construct_WiderFace(constructor, self)
