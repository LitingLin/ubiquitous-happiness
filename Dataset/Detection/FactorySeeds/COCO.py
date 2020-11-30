from Dataset._base_seed import BaseSeed
from Dataset.DataSplit import DataSplit
from enum import Flag, auto


class COCOVersion(Flag):
    _2014 = auto()
    _2017 = auto()


class COCO_Seed(BaseSeed):
    def __init__(self, root_path=None, data_split=DataSplit.Training | DataSplit.Validation, version: COCOVersion = COCOVersion._2014, include_crowd=False):
        if root_path is None:
            root_path = self._getPathFromConfig('COCO_PATH')
        if version == COCOVersion._2014:
            name = 'COCO2014'
        elif version == COCOVersion._2017:
            name = 'COCO2017'
        else:
            raise Exception
        if not include_crowd:
            name += '-nocrowd'
        super(COCO_Seed, self).__init__(name, root_path, data_split, 2)
        self.version = version
        self.include_crowd = include_crowd

    def construct(self, constructor):
        from ..Sources.COCO import construct_COCO
        construct_COCO(constructor, self)
