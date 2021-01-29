from enum import Flag, auto
from Dataset.Base.factory_seed import BaseSeed
from Dataset.Type.data_split import DataSplit


class NFSDatasetVersionFlag(Flag):
    fps_30 = auto()
    fps_240 = auto()


class NFS_Seed(BaseSeed):
    def __init__(self, root_path: str=None, version: NFSDatasetVersionFlag = NFSDatasetVersionFlag.fps_240, manual_anno_only=True):
        if root_path is None:
            root_path = self.get_path_from_config('NFS_PATH')
        if version == NFSDatasetVersionFlag.fps_30:
            name = 'NFS-fps_30'
        elif version == NFSDatasetVersionFlag.fps_240:
            name = 'NFS-fps_240'
        else:
            raise Exception
        if manual_anno_only:
            name += '-no_generated'
        self.nfs_version = version
        self.manual_anno_only = manual_anno_only
        super(NFS_Seed, self).__init__(name, root_path, DataSplit.Full, 1)

    def construct(self, constructor):
        from .Impl.NFS import construct_NFS
        construct_NFS(constructor, self)
