from miscellanies.yaml_ops import yaml_load
import os


def get_path_from_config(name: str):
    config_folder = os.path.dirname(__file__)
    config_path = os.path.join(config_folder, 'path.yaml')
    if not os.path.exists(config_path):
        import shutil
        shutil.copyfile(os.path.join(config_folder, 'path.template.yaml'), config_path)
        raise RuntimeError('Setup the paths in Dataset/Config/path.yaml first')
    config = yaml_load(config_path)
    return config[name]
