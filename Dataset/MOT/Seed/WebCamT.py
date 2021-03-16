from Dataset.Base.factory_seed import BaseSeed
from Dataset.Type.data_split import DataSplit


class WebCamT_Seed(BaseSeed):
    def __init__(self, root_path: str=None, data_split=DataSplit.Training | DataSplit.Validation, include_passenger: bool=False):
        if root_path is None:
            root_path = self.get_path_from_config('WebCamT_PATH')
        name = 'WebCamT'
        if not include_passenger:
            name += '-exclude_passenger'
        super(WebCamT_Seed, self).__init__(name, root_path, data_split, 2)
        self.include_passenger = include_passenger

    def construct(self, constructor):
        from .Impl.WebCamT import construct_WebCamT
        construct_WebCamT(constructor, self)
