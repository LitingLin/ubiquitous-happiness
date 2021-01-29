from Dataset.Base.factory_seed import BaseSeed
from Dataset.Type.data_split import DataSplit
import os


class VisDrone2019_DET_Seed(BaseSeed):
    def __init__(self, root_path: str=None, data_split=DataSplit.Training):
        if root_path is None:
            if data_split == DataSplit.Training:
                root_path = self.get_path_from_config('VisDrone_2019_DET_Training_PATH')
            elif data_split == DataSplit.Validation:
                root_path = self.get_path_from_config('VisDrone_2019_DET_Val_PATH')
            elif data_split == DataSplit.Testing:
                root_path = self.get_path_from_config('VisDrone_2019_DET_Test_PATH')
            else:
                raise ValueError('Unsupported data type')
        images_path = os.path.join(root_path, 'images')
        annotation_path = os.path.join(root_path, 'annotations')

        super(VisDrone2019_DET_Seed, self).__init__('VisDrone-2019-DET', images_path, data_split, 1)
        self.annotation_path = annotation_path

    def construct(self, constructor):
        from .Impl.VisDrone_2019_DET import construct_VisDrone2019_DET
        construct_VisDrone2019_DET(constructor, self)
