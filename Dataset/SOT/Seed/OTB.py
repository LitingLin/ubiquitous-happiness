from Dataset.Base.factory_seed import BaseSeed
from Dataset.Type.data_split import DataSplit
from enum import Enum, auto


class OTB100Split(Enum):
    OTB100 = auto()
    OTB50 = auto()
    OTB2013 = auto()
    OTB2015 = auto()


class OTB_Seed(BaseSeed):
    def __init__(self, root_path: str=None, split: OTB100Split=OTB100Split.OTB100):
        if root_path is None:
            root_path = self.get_path_from_config('OTB_PATH')

        self.otb_split = split

        super(OTB_Seed, self).__init__(split.name, root_path, DataSplit.Full, 3)

    def construct(self, constructor):
        from .Impl.OTB import construct_OTB
        construct_OTB(constructor, self)
