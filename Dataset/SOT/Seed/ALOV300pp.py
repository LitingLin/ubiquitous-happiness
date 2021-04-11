from Dataset.Base.factory_seed import BaseSeed
from Dataset.Type.data_split import DataSplit


class ALOV300pp_Seed(BaseSeed):
    def __init__(self, root_path: str=None, annotation_path: str=None):
        if root_path is None:
            root_path = self.get_path_from_config('ALOV300++_Frames_PATH')
        if annotation_path is None:
            self.annotation_path = self.get_path_from_config('ALOV300++_Annotation_PATH')
        super(ALOV300pp_Seed, self).__init__('ALOV300++', root_path, DataSplit.Full, 1)

    def construct(self, constructor):
        from .Impl.ALOV300pp import construct_ALOV300pp
        construct_ALOV300pp(constructor, self)
