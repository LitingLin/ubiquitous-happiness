from Dataset._base_seed import BaseSeed
from Dataset.DataSplit import DataSplit
import os


class CityPersons_Seed(BaseSeed):
    def __init__(self, cityscapes_path: str=None, annotation_path: str=None, data_split: DataSplit = DataSplit.Training | DataSplit.Validation, things_only: bool = True):
        name = 'citypersons'
        if things_only:
            name += '-things_only'
        cityscapes_path = os.path.join(cityscapes_path, 'leftImg8bit')
        if cityscapes_path is None:
            cityscapes_path = self._getPathFromConfig('Cityscapes_PATH')
        if annotation_path is None:
            annotation_path = self._getPathFromConfig('CityPersons_Annotation_PATH')
        super(CityPersons_Seed, self).__init__(name, cityscapes_path, data_split, 1)
        self.annotation_path = annotation_path
        self.things_only = things_only

    def construct(self, constructor):
        from ..Sources.CityPersons import construct_cityPersons
        construct_cityPersons(constructor, self)
