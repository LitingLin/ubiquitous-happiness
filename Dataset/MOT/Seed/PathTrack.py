# https://www.trace.ethz.ch/publications/2017/pathtrack/index.html

from Dataset.Base.factory_seed import BaseSeed
from Dataset.Type.data_split import DataSplit


class PathTrack_Seed(BaseSeed):
    def __init__(self, root_path: str=None, data_split: DataSplit=DataSplit.Training | DataSplit.Validation):
        name = 'PathTrack'
        if root_path is None:
            root_path = self.get_path_from_config('PathTrack_PATH')
        super(PathTrack_Seed, self).__init__(name, root_path, data_split, 1)

    def construct(self, constructor):
        from .Impl.PathTrack import construct_PathTrack
        construct_PathTrack(constructor, self)
