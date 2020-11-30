from Dataset._base_seed import BaseSeed
from Dataset.DataSplit import DataSplit
import os


class BDD100k_Images_Seed(BaseSeed):
    def __init__(self, root_path: str=None, labels_path:str=None, data_split=DataSplit.Training | DataSplit.Validation):
        if root_path is None:
            root_path = self._getPathFromConfig('BDD100k_PATH')
        root_path = os.path.join(root_path, 'images', '100k')
        super(BDD100k_Images_Seed, self).__init__('BDD100k-Images', root_path, data_split, 1)
        self.labels_path = labels_path

    def construct(self, constructor):
        from ..Sources.BDD100k_Images import construct_BDD100k_Images
        construct_BDD100k_Images(constructor, self)
