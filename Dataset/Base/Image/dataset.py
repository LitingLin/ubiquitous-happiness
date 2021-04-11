import os
from Dataset.Type.specialized_dataset import SpecializedImageDatasetType
from data.types.bounding_box_format import BoundingBoxFormat
from Dataset.Base.Common.dataset import _BaseDataset, _BaseDatasetObject
from Miscellaneous.platform_style_path import join_path

__version__ = 1


class ImageDatasetImage:
    def __init__(self, image: dict, root_path: str):
        self.image = image
        self.root_path = root_path

    def has_attribute(self, name: str):
        return name in self.image

    def get_attribute(self, name: str):
        return self.image[name]

    def get_all_attribute_name(self):
        return self.image.keys()

    def get_image_file_name(self):
        return os.path.basename(self.image['path'])

    def get_image_path(self):
        return join_path(self.root_path, self.image['path'])

    def get_image_size(self):
        return self.image['size']

    def __len__(self):
        if 'objects' not in self.image:
            return 0
        return len(self.image['objects'])

    def __getitem__(self, index: int):
        return _BaseDatasetObject(self.image['objects'][index])


class ImageDataset(_BaseDataset):
    def __init__(self, root_path: str, dataset: dict=None):
        super(ImageDataset, self).__init__(root_path, dataset)

    @staticmethod
    def load(yaml_path: str, root_path: str):
        return ImageDataset(root_path, _BaseDataset.load(yaml_path, __version__))

    def get_constructor(self, type_: SpecializedImageDatasetType, version: int):
        assert isinstance(type_, SpecializedImageDatasetType)
        if type_ == SpecializedImageDatasetType.Detection:
            from Dataset.DET.Constructor.base import DetectionDatasetConstructorGenerator
            return DetectionDatasetConstructorGenerator(self.dataset, self.root_path, version)
        else:
            raise NotImplementedError

    def specialize(self, type_: SpecializedImageDatasetType, path: str):
        assert isinstance(type_, SpecializedImageDatasetType)
        if type_ == SpecializedImageDatasetType.Detection:
            from Dataset.DET.Storage.MemoryMapped.constructor import construct_detection_dataset_memory_mapped_from_base_image_dataset
            from Dataset.DET.Storage.MemoryMapped.dataset import DetectionDataset_MemoryMapped
            return DetectionDataset_MemoryMapped(self.root_path,
                                                 construct_detection_dataset_memory_mapped_from_base_image_dataset(
                                                     self.dataset, path))
        else:
            raise NotImplementedError

    def get_manipulator(self):
        from .manipulator import ImageDatasetManipulator
        return ImageDatasetManipulator(self.dataset)

    def __getitem__(self, index: int):
        return ImageDatasetImage(self.dataset['images'][index], self.root_path)

    def __len__(self):
        return len(self.dataset['images'])
