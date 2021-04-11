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
        from Miscellaneous.yaml_ops import yaml_load
        import os
        config_folder = os.path.join(os.path.dirname(__file__), '..', 'Config')
        config_path = os.path.join(config_folder, 'path.yaml')
        if not os.path.exists(config_path):
            import shutil
            shutil.copyfile(os.path.join(config_folder, 'path.template.yaml'), config_path)
            raise RuntimeError('Setup the paths in Dataset/Config/path.yaml first')
        config = yaml_load(config_path)
        return config[name]
