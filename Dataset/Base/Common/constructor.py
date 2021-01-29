import os
from Dataset.Type.data_split import DataSplit
from PIL import Image
from tqdm import tqdm
import Dataset.Base.Video.dataset
import Dataset.Base.Image.dataset

image_dataset_key_exclude_list = ('name', 'split', 'version', 'filters', 'type', 'category_id_name_map', 'images')
image_dataset_image_key_exclude_list = ('size', 'path', 'objects')
image_dataset_object_key_exclude_list = ('category_id', 'bounding_box')

video_dataset_key_exclude_list = ('name', 'split', 'version', 'filters', 'type', 'category_id_name_map', 'sequences')
video_dataset_sequence_key_exclude_list = ('name', 'path', 'fps', 'frames', 'objects')
video_dataset_frame_key_exclude_list = ('path', 'size', 'objects')
video_dataset_sequence_object_key_exclude_list = ('category_id', 'id')
video_dataset_frame_object_key_exclude_list = ('id', 'bounding_box')


class DatasetConstructorProcessBar:
    def __init__(self):
        self.pbar = None
        self.total = None
        self.dataset_name = None
        self.dataset_split = None
        self.sequence_name = None

    def _construct_bar_if_not_exists(self):
        if self.pbar is None:
            self.pbar = tqdm(total=self.total)
            self._update_pbar_desc()

    def set_total(self, total: int):
        assert isinstance(total, int)
        self.total = total

    def _update_pbar_desc(self):
        if self.pbar is None:
            return
        assert self.dataset_name is not None
        string = self.dataset_name
        if self.dataset_split is not None:
            string += f'({self.dataset_split})'
        self.pbar.set_description_str(string)
        if self.sequence_name is not None:
            self.pbar.set_postfix_str(self.sequence_name)

    def set_dataset_name(self, name: str):
        self.dataset_name = name
        self._update_pbar_desc()

    def set_dataset_split(self, split: DataSplit):
        if split == DataSplit.Full:
            self.dataset_split = None
            return
        self.dataset_split = split.name
        self._update_pbar_desc()

    def set_sequence_name(self, name: str):
        self.sequence_name = name
        self._update_pbar_desc()

    def update(self, n=1):
        self._construct_bar_if_not_exists()
        self.pbar.update(n)

    def close(self):
        if self.pbar is not None:
            self.pbar.close()
            self.pbar = None


def _root_path_impl(root_path):
    return os.path.abspath(root_path)


def _add_path_impl(path, root_path):
    path = os.path.abspath(path)
    rel_path = os.path.relpath(path, root_path)
    return rel_path


def set_path_(image_dict: dict, image_path: str, root_path: str, size):
    image_dict['path'] = _add_path_impl(image_path, root_path)
    if size is None:
        image = Image.open(image_path)
        image_dict['size'] = image.size
    else:
        assert len(size) == 2
        for v in size:
            if isinstance(v, float):
                assert v.is_integer()
        size = tuple([int(v) for v in size])
        image_dict['size'] = size


def generate_sequence_path_(sequence: dict):
    assert 'path' not in sequence
    paths = []
    for frame in sequence['frames']:
        paths.append(frame['path'])
    sequence_path = os.path.commonpath(paths)
    sequence['path'] = sequence_path
    for frame in sequence['frames']:
        frame['path'] = os.path.relpath(frame['path'], sequence_path)


class BaseDatasetSequenceConstructor:
    def __init__(self, sequence: dict, root_path: str, pbar: DatasetConstructorProcessBar):
        self.sequence = sequence
        self.root_path = root_path
        self.pbar = pbar

    def set_name(self, name: str):
        self.sequence['name'] = name
        self.pbar.set_sequence_name(name)

    def set_fps(self, fps):
        self.sequence['fps'] = fps

    def set_attribute(self, name, value):
        self.sequence[name] = value

    def merge_attributes(self, attributes):
        for key, value in attributes.items():
            self.sequence[key] = value


class BaseDatasetImageConstructor:
    def __init__(self, image: dict, root_path: str):
        self.image = image
        self.root_path = root_path

    def set_attribute(self, name: str, value):
        self.image[name] = value

    def set_path(self, path: str, image_size=None):
        set_path_(self.image, path, self.root_path, image_size)

    def get_image_size(self):
        return self.image['size']

    def merge_attributes(self, attributes):
        for key, value in attributes.items():
            self.image[key] = value


class BaseDatasetImageConstructorGenerator:
    def __init__(self, pbar):
        self.pbar = pbar

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.pbar.update()


class BaseDatasetSequenceConstructorGenerator:
    def __init__(self, sequence: dict, pbar):
        self.sequence = sequence
        self.pbar = pbar

    def __exit__(self, exc_type, exc_val, exc_tb):
        generate_sequence_path_(self.sequence)
        self.pbar.update()


class _BaseDatasetConstructor:
    def __init__(self, dataset: dict, root_path: str, dataset_type: str, schema_version: int, data_version: int, pbar: DatasetConstructorProcessBar):
        if 'type' in dataset:
            assert dataset['type'] == dataset_type
        else:
            dataset['type'] = dataset_type
        if 'version' in dataset:
            assert dataset['version'][0] == schema_version
            assert dataset['version'][1] == data_version
        else:
            dataset['version'] = [schema_version, data_version]

        assert 'filters' not in dataset or len(dataset['filters']) == 0

        self.dataset = dataset
        self.root_path = _root_path_impl(root_path)
        self.pbar = pbar

    def set_category_id_name_map(self, category_id_name_map: dict):
        self.dataset['category_id_name_map'] = category_id_name_map

    def set_name(self, name: str):
        self.dataset['name'] = name
        self.pbar.set_dataset_name(name)

    def set_split(self, split: DataSplit):
        assert len(split.name) > 0
        self.dataset['split'] = split.name
        self.pbar.set_dataset_split(split)


class BaseVideoDatasetConstructor(_BaseDatasetConstructor):
    def __init__(self, dataset: dict, root_path: str, data_version: int, pbar: DatasetConstructorProcessBar):
        super(BaseVideoDatasetConstructor, self).__init__(dataset, root_path, 'video', Dataset.Base.Video.dataset.__version__, data_version, pbar)

    def set_total_number_of_sequences(self, number: int):
        self.pbar.set_total(number)

    def set_attribute(self, name: str, value):
        self.dataset[name] = value


class BaseImageDatasetConstructor(_BaseDatasetConstructor):
    def __init__(self, dataset: dict, root_path: str, data_version: int, pbar: DatasetConstructorProcessBar):
        super(BaseImageDatasetConstructor, self).__init__(dataset, root_path, 'image', Dataset.Base.Image.dataset.__version__, data_version, pbar)

    def set_total_number_of_images(self, number: int):
        self.pbar.set_total(number)

    def set_attribute(self, name: str, value):
        self.dataset[name] = value


class BaseDatasetConstructorGenerator:
    def __init__(self, dataset: dict, root_path: str, version: int, constructor_type):
        self.dataset = dataset
        self.root_path = root_path
        self.version = version
        self.constructor_type = constructor_type
        self.pbar = DatasetConstructorProcessBar()

    def __enter__(self):
        return self.constructor_type(self.dataset, self.root_path, self.version, self.pbar)

    def __exit__(self, exc_type, exc_val, exc_tb):
        assert 'split' in self.dataset
        assert 'name' in self.dataset
        self.pbar.close()
