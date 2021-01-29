from Dataset.Type.data_split import DataSplit
from Dataset.Base.factory_seed import BaseSeed


class HMDB51_Seed(BaseSeed):
    def __init__(self, root_path: str=None, annotation_path: str=None):
        if root_path is None:
            root_path = self.get_path_from_config('HMDB51_PATH')
        if annotation_path is None:
            annotation_path = self.get_path_from_config('HMDB51_ANNOTATION_PATH')
        super(HMDB51_Seed, self).__init__('HMDB51', root_path, DataSplit.Full, 3)
        self.annotation_path = annotation_path

    def construct(self, constructor):
        from .Impl.HMDB51 import construct_HMDB51
        construct_HMDB51(constructor, self)
