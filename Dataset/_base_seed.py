from Dataset.DataSplit import DataSplit


class BaseSeed:
    name: str
    root_path: str
    data_split: DataSplit
    data_version: int

    def __init__(self, name, root_path: str, data_split: DataSplit, data_version: int):
        self.name = name
        self.root_path = root_path
        self.data_split = data_split
        self.data_version = data_version

    def _getPathFromConfig(self, name: str):
        try:
            from Dataset.Config.path import DatasetPath
        except ImportError:
            import shutil
            import os
            config_folder = os.path.join(os.path.dirname(__file__), 'Config')
            shutil.copyfile(os.path.join(config_folder, 'path.template.py'), os.path.join(config_folder, 'path.py'))
            raise Exception('Setup the paths in Dataset/Config/path.py first')
        return getattr(DatasetPath, name)
