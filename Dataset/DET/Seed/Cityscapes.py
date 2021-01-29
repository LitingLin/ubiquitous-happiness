from enum import Flag, auto
from Dataset.Base.factory_seed import BaseSeed
from Dataset.Type.data_split import DataSplit


class CityscapesAnnotationSource(Flag):
    PreferFine = auto()
    FineOnly = auto()
    CoarseOnly = auto()


class Cityscapes_Seed(BaseSeed):
    def __init__(self, root_path: str=None,
                 data_split=DataSplit.Training | DataSplit.Validation,
                 annotation_source=CityscapesAnnotationSource.PreferFine, things_only=True):
        if root_path is None:
            root_path = self.get_path_from_config('Cityscapes_PATH')
        name = 'Cityscapes'

        if things_only:
            name += '-things_only'
        if annotation_source == CityscapesAnnotationSource.PreferFine:
            name += '-prefer_fine'
        elif annotation_source == CityscapesAnnotationSource.FineOnly:
            name += '-fine_only'
        elif annotation_source == CityscapesAnnotationSource.CoarseOnly:
            name += '-coarse_only'
        else:
            raise Exception
        super(Cityscapes_Seed, self).__init__(name, root_path, data_split, 2)
        self.annotation_source = annotation_source
        self.things_only = things_only

    def construct(self, constructor):
        from .Impl.Cityscapes import construct_CityScapes
        construct_CityScapes(constructor, self)
