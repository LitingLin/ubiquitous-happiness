from Dataset.DataSplit import DataSplit
from Dataset._base_seed import BaseSeed


class Olympic_Sports_Seed(BaseSeed):
    def __init__(self, root_path: str=None, annotation_path: str=None):
        if root_path is None:
            root_path = self._getPathFromConfig('Olympic_Sports_PATH')
        if annotation_path is None:
            annotation_path = self._getPathFromConfig('Olympic_Sports_Annotation_PATH')
        super(Olympic_Sports_Seed, self).__init__('OlympicSports', root_path, DataSplit.Full, 1)
        self.annotation_path = annotation_path

    def construct(self, constructor):
        from ..Sources.Olympic_Sports import construct_OlympicSports
        construct_OlympicSports(constructor, self)
