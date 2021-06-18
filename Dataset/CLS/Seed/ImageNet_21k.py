from Dataset.Base.factory_seed import BaseSeed
from Dataset.Type.data_split import DataSplit


class ImageNet_21k_Seed(BaseSeed):
    def __init__(self, root_path: str=None):
        if root_path is None:
            root_path = self.get_path_from_config('ImageNet-21k_PATH')
        super(ImageNet_21k_Seed, self).__init__('ImageNet-21k', root_path, DataSplit.Full, 1)

    def construct(self, constructor):
        from .Impl.ImageNet_21k import construct_ImageNet_21k
        construct_ImageNet_21k(constructor, self)
