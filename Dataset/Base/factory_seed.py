from Dataset.Type.data_split import DataSplit


class BaseSeed:
    name: str
    root_path: str
    data_split: DataSplit
    version: int

    def __init__(self, name, root_path: str, data_split: DataSplit, version: int):
        assert root_path is not None and len(root_path) > 0
        self.name = name
        self.root_path = root_path
        self.data_split = data_split
        self.version = version

    @staticmethod
    def get_path_from_config(name: str):
        import Dataset.Config.path
        return Dataset.Config.path.get_path_from_config(name)
