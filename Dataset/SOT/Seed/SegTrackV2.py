# http://cpl.cc.gatech.edu/projects/SegTrack/
from Dataset.Base.factory_seed import BaseSeed
from Dataset.Type.data_split import DataSplit


class SegTrackV2_Seed(BaseSeed):
    def __init__(self, root_path: str=None):
        if root_path is None:
            root_path = self.get_path_from_config('SegTrackV2_PATH')
        super(SegTrackV2_Seed, self).__init__('SegTrackV2', root_path, DataSplit.Full, 1)

    def construct(self, constructor):
        from .Impl.SegTrackV2 import construct_SegTrackV2
        construct_SegTrackV2(constructor, self)
