from Dataset.Base.factory_seed import BaseSeed
from Dataset.Type.data_split import DataSplit
import os


class VisDrone2019_SOT_Seed(BaseSeed):
    def __init__(self, root_path: str=None, data_split: DataSplit = DataSplit.Training):
        if root_path is None:
            if data_split == DataSplit.Training:
                root_path = self.get_path_from_config('VisDrone_2019_SOT_Training_PATH')
            elif data_split == DataSplit.Validation:
                root_path = self.get_path_from_config('VisDrone_2019_SOT_Val_PATH')
            elif data_split == DataSplit.Testing:
                root_path = self.get_path_from_config('VisDrone_2019_SOT_Test_PATH')
            else:
                raise ValueError('Unsupported data type')
        sequences_path = os.path.join(root_path, 'sequences')
        annotations_path = os.path.join(root_path, 'annotations')
        super(VisDrone2019_SOT_Seed, self).__init__('VisDrone2019-SOT', sequences_path, data_split, 1)
        self.annotations_path = annotations_path

    def construct(self, constructor):
        from .Impl.VisDrone_2019_SOT import construct_VisDrone_2019_SOT
        construct_VisDrone_2019_SOT(constructor, self)
