from Miscellaneous.yaml_ops import yaml_dump, yaml_load
from Dataset.Type.incompatible_error import IncompatibleError
from Dataset.Base.Common.Operator.filters import filter_list_deserialize
from Dataset.Type.bounding_box_format import BoundingBoxFormat
from Dataset.Base.Common.ops import get_bounding_box, get_bounding_box_in_format


class _BaseDatasetObject:
    def __init__(self, object_: dict):
        self.object_ = object_

    def get_bounding_box(self):
        return get_bounding_box(self.object_)

    def get_bounding_box_in_format(self, format_: BoundingBoxFormat=BoundingBoxFormat.XYWH):
        return get_bounding_box_in_format(self.object_, format_)

    def get_category_id(self):
        return self.object_['category_id']

    def has_category_id(self):
        return 'category_id' in self.object_

    def has_bounding_box(self):
        return 'bounding_box' in self.object_

    def has_id(self):
        return 'id' in self.object_

    def get_id(self):
        return self.object_['id']

    def has_attribute(self, name: str):
        return name in self.object_

    def get_attribute(self, name: str):
        return self.object_[name]

    def get_all_attribute_name(self):
        return self.object_.keys()


class _BaseDataset:
    dataset: dict

    def __init__(self, root_path: str, dataset: dict=None):
        self.dataset = dataset
        self.root_path = root_path

    def get_root_path(self):
        return self.root_path

    def set_root_path(self, path: str):
        self.root_path = path

    def get_version(self):
        return self.dataset['version'][1]

    @staticmethod
    def load(path: str, scheme_version):
        import pickle
        with open(path, 'rb') as f:
            dataset = pickle.load(f)
        if dataset['version'][0] != scheme_version:
            raise IncompatibleError
        return dataset

    def dump(self, path: str):
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self.dataset, f)

    @staticmethod
    def load_yaml(yaml_path: str, scheme_version):
        dataset = yaml_load(yaml_path)
        if dataset['version'][0] != scheme_version:
            raise IncompatibleError
        return dataset

    def dump_yaml(self, yaml_path: str):
        yaml_dump(self.dataset, yaml_path)

    def get_applied_filters(self):
        if 'filters' in self.dataset:
            return filter_list_deserialize(self.dataset['filters'])
        return None

    def has_attribute(self, name: str):
        return name in self.dataset

    def get_attribute(self, name: str):
        return self.dataset[name]

    def get_all_attribute_name(self):
        return self.dataset.keys()

    def has_category_id_name_map(self):
        return 'category_id_name_map' in self.dataset

    def get_category_id_name_map(self):
        return self.dataset['category_id_name_map']

    def get_category_name_by_id(self, id_):
        return self.dataset['category_id_name_map'][id_]

    def get_name(self):
        return self.dataset['name']
