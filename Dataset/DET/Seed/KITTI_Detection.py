from Dataset.Base.factory_seed import BaseSeed
from Dataset.Type.data_split import DataSplit
import os


class KITTI_Detection_Seed(BaseSeed):
    def __init__(self, root_path: str=None, data_split: DataSplit = DataSplit.Training, exclude_dontcare=True):
        if root_path is None:
            root_path = self.get_path_from_config('KITTI_Detection_PATH')

        name = 'KITTI-Detection'
        if exclude_dontcare:
            name += '-no_dontcare'
        images_root_path = os.path.join(root_path, 'training', 'image_2')
        super(KITTI_Detection_Seed, self).__init__(name, images_root_path, data_split, 3)
        self.exclude_dontcare = exclude_dontcare

    def construct(self, constructor):
        from .Impl.KITTI import construct_KITTI_Detection
        construct_KITTI_Detection(constructor, self)
