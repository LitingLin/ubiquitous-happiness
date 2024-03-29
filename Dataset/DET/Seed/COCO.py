from Dataset.Base.factory_seed import BaseSeed
from Dataset.Type.data_split import DataSplit
from enum import Flag, auto


class COCOVersion(Flag):
    _2014 = auto()
    _2017 = auto()


class COCO_Seed(BaseSeed):
    def __init__(self, root_path=None, data_split=DataSplit.Training | DataSplit.Validation, version: COCOVersion = COCOVersion._2014, include_crowd=False):
        if version == COCOVersion._2014:
            name = 'COCO2014'
            if root_path is None:
                root_path = self.get_path_from_config('COCO_2014_PATH')
        elif version == COCOVersion._2017:
            name = 'COCO2017'
            if root_path is None:
                root_path = self.get_path_from_config('COCO_2017_PATH')
        else:
            raise Exception
        if not include_crowd:
            name += '-nocrowd'
        super(COCO_Seed, self).__init__(name, root_path, data_split, 2)
        self.coco_version = version
        self.include_crowd = include_crowd

    def construct(self, constructor):
        from .Impl.COCO import construct_COCO
        construct_COCO(constructor, self)
