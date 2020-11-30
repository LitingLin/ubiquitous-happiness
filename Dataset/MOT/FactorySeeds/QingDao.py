from Dataset._base_seed import BaseSeed
from enum import IntFlag, auto
from Dataset.DataSplit import DataSplit


class QingDaoDataset_SceneTypes(IntFlag):
    DianJing = auto()
    LuKou = auto()
    GaoDian = auto()
    Full = DianJing | LuKou | GaoDian

    def __str__(self):
        if self.value == QingDaoDataset_SceneTypes.Full:
            return 'full'
        string = ''
        if self.value & QingDaoDataset_SceneTypes.DianJing:
            string += 'dianjing'
        if self.value & QingDaoDataset_SceneTypes.LuKou:
            string += 'lukou'
        if self.value & QingDaoDataset_SceneTypes.GaoDian:
            string += 'gaodian'
        return string


class QingDao_Seed(BaseSeed):
    def __init__(self, root_path: str=None, scene_type: QingDaoDataset_SceneTypes=QingDaoDataset_SceneTypes.Full):
        name = 'QingDao-' + str(scene_type)
        if root_path is None:
            root_path = self._getPathFromConfig('QingDao_PATH')
        super(QingDao_Seed, self).__init__(name, root_path, DataSplit.Full, 1)
        self.scene_type = scene_type

    def construct(self, constructor):
        from ..Sources.QingDao import construct_QingDao
        construct_QingDao(constructor, self)
