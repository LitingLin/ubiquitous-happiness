import yaml
from yaml import CSafeLoader as Loader, CSafeDumper as Dumper


__all__ = ['yaml_load', 'yaml_dump']


def yaml_load(path: str):
    with open(path, 'rb') as f:
        object_ = yaml.load(f, Loader=Loader)
    return object_


def yaml_dump(object_, path: str):
    with open(path, 'wb') as f:
        yaml.dump(object_, f, encoding='utf-8', default_flow_style=False, Dumper=Dumper)
