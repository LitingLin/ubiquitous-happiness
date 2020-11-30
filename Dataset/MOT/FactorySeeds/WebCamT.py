from Dataset._base_seed import BaseSeed
from Dataset.DataSplit import DataSplit


class WebCamT_Seed(BaseSeed):
    def __init__(self, root_path: str=None, data_split=DataSplit.Training | DataSplit.Validation, include_passenger: bool=False):
        if root_path is None:
            root_path = self._getPathFromConfig('WebCamT_PATH')
        name = 'WebCamT'
        if not include_passenger:
            name += '-exclude_passenger'
        super(WebCamT_Seed, self).__init__(name, root_path, data_split, 2)
        self.include_passenger = include_passenger

    def construct(self, constructor):
        from ..Sources.WebCamT import construct_WebCamT
        construct_WebCamT(constructor, self)
