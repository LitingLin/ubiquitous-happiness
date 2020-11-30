from Dataset._base_seed import BaseSeed
from Dataset.DataSplit import DataSplit


class OpenImages_Seed(BaseSeed):
    def __init__(self, root_path: str=None, data_split=DataSplit.Full):
        if root_path is None:
            root_path = self._getPathFromConfig('Open_Images_PATH')
        super(OpenImages_Seed, self).__init__('Open-Images-V5', root_path, data_split, 2)

    def construct(self, constructor):
        from ..Sources.OpenImages import construct_OpenImages
        construct_OpenImages(constructor, self)
