import pathlib
from Dataset.DataSplit import DataSplit
from Dataset.CacheService.common import tryLoadCache, makeCache


class DatasetConstructor_CacheService_Base:
    def __init__(self, dataset):
        self.dataset = dataset
        self.root_path = None

    def setRootPath(self, root_path: str):
        root_path = pathlib.Path(root_path)
        assert root_path.exists()
        self.root_path = root_path
        self.dataset.root_path = str(root_path)

    def setDataVersion(self, version: int):
        self.dataset.data_version = version

    def setDataSplit(self, data_split: DataSplit):
        self.dataset.data_split = data_split

    def tryLoadCache(self):
        assert self.dataset.data_version is not None
        assert self.dataset.name is not None
        if tryLoadCache(self.dataset):
            self.dataset.root_path = str(self.root_path)
            return True
        else:
            return False

    def makeCache(self):
        makeCache(self.dataset)

    def setDatasetName(self, name: str):
        self.dataset.name = name
