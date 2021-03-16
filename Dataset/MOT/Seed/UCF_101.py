from Dataset.Type.data_split import DataSplit
from Dataset.Base.factory_seed import BaseSeed


class UCF101_Seed(BaseSeed):
    def __init__(self, root_path: str=None, annotation_file_path: str=None):
        if root_path is None:
            root_path = self.get_path_from_config('UCF101_PATH')
        if annotation_file_path is None:
            annotation_file_path = self.get_path_from_config('UCF101_THUMOS_2015_HUMAN_ANNOTATION')
        super(UCF101_Seed, self).__init__('UCF-101-THUMOS', root_path, DataSplit.Full, 2)
        self.annotation_file_path = annotation_file_path

    def construct(self, constructor):
        from .Impl.UCF_101 import construct_UCF101THUMOSDataset
        construct_UCF101THUMOSDataset(constructor, self)
