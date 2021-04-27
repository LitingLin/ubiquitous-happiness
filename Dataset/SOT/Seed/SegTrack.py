# http://cpl.cc.gatech.edu/projects/SegTrack/
from Dataset.Base.factory_seed import BaseSeed
from Dataset.Type.data_split import DataSplit


class SegTrack_Seed(BaseSeed):
    def __init__(self, root_path: str=None):
        if root_path is None:
            root_path = self.get_path_from_config('SegTrack_PATH')
        super(SegTrack_Seed, self).__init__('SegTrack', root_path, DataSplit.Full, 1)

    def construct(self, constructor):
        from .Impl.SegTrack import construct_SegTrack
        construct_SegTrack(constructor, self)
