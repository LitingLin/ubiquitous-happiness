from Dataset.Base.factory_seed import BaseSeed
from Dataset.Type.data_split import DataSplit
import os


class KITTI_Tracking_Seed(BaseSeed):
    def __init__(self, image_path: str=None, label_path: str=None):
        if image_path is None:
            image_path = self.get_path_from_config('KITTI_Tracking_Image_PATH')
        if label_path is None:
            label_path = self.get_path_from_config('KITTI_Tracking_Label_PATH')

        image_path = os.path.join(image_path, 'training', 'image_02')
        super(KITTI_Tracking_Seed, self).__init__('KITTI-Object-Tracking', image_path, DataSplit.Training, 1)
        self.label_path = label_path

    def construct(self, constructor):
        from .Impl.KITTI_Tracking import construct_KITTI_Tracking
        construct_KITTI_Tracking(constructor, self)
