from enum import Flag, auto
from Dataset._base_seed import BaseSeed
from Dataset.DataSplit import DataSplit


class NFSDatasetVersionFlag(Flag):
    fps_30 = auto()
    fps_240 = auto()


class NFS_Seed(BaseSeed):
    def __init__(self, root_path: str=None, version: NFSDatasetVersionFlag = NFSDatasetVersionFlag.fps_30):
        if root_path is None:
            root_path = self._getPathFromConfig('NFS_PATH')
        if version == NFSDatasetVersionFlag.fps_30:
            name = 'NFS-fps_30'
        elif version == NFSDatasetVersionFlag.fps_240:
            name = 'NFS-fps_240'
        else:
            raise Exception
        self.version = version
        super(NFS_Seed, self).__init__(name, root_path, DataSplit.Full, 1)

    def construct(self, constructor):
        from ..Sources.NFS import construct_NFS
        construct_NFS(constructor, self)
