from Dataset.Base.Common.ops import get_bounding_box, set_bounding_box_, bounding_box_convert_format_, bounding_box_fit_in_image_size_, bounding_box_update_validity_, get_bounding_box_in_format
from Dataset.Type.bounding_box_format import BoundingBoxFormat


class ImageDatasetObjectManipulator:
    def __init__(self, image: dict, index_of_object: int, parent_iterator=None):
        self.object_ = image['objects'][index_of_object]
        self.parent_image = image
        self.index_of_object = index_of_object
        self.parent_iterator = parent_iterator

    def get_bounding_box(self):
        return get_bounding_box(self.object_)

    def set_bounding_box(self, bounding_box, bounding_box_format: BoundingBoxFormat, validity=None):
        set_bounding_box_(self.object_, bounding_box, bounding_box_format, validity)

    def bounding_box_mark_validity(self, value: bool):
        self.object_['bounding_box']['validity'] = value

    def bounding_box_convert_format(self, target_format: BoundingBoxFormat, strict = True):
        bounding_box_convert_format_(self.object_, target_format, strict)

    def bounding_box_fit_in_image_size(self):
        bounding_box_fit_in_image_size_(self.object_, self.parent_image)

    def bounding_box_update_validity(self, skip_if_mark_non_validity = True):
        bounding_box_update_validity_(self.object_, self.parent_image, skip_if_mark_non_validity)

    def get_bounding_box_with_format(self, bounding_box_format: BoundingBoxFormat, strict = True):
        return get_bounding_box_in_format(self.object_, bounding_box_format, strict)

    def delete_bounding_box(self):
        del self.object_['bounding_box']

    def get_category_id(self):
        return self.object_['category_id']

    def has_category_id(self):
        return 'category_id' in self.object_

    def has_bounding_box(self):
        return 'bounding_box' in self.object_

    def delete_category_id(self):
        del self.object_['category_id']

    def set_category_id(self, id_: int):
        self.object_['category_id'] = id_

    def get_attribute(self, name: str):
        return self.object_[name]

    def delete(self):
        del self.parent_image['objects'][self.index_of_object]
        del self.object_
        if self.parent_iterator is not None:
            self.parent_iterator.deleted()


class ImageDatasetObjectManipulatorIterator:
    def __init__(self, frame: dict):
        self.frame = frame
        self.index = 0

    def __next__(self):
        if self.index >= len(self.frame['objects']):
            raise StopIteration

        modifier = ImageDatasetObjectManipulator(self.frame, self.index, self)
        self.index += 1
        return modifier

    def deleted(self):
        self.index -= 1


class ImageDatasetImageManipulator:
    def __init__(self, dataset: dict, index_of_image: int, parent_iterator=None):
        self.image = dataset['images'][index_of_image]
        self.dataset = dataset
        self.index_of_image = index_of_image
        self.parent_iterator = parent_iterator

    def get_image_size(self):
        return self.image['size']

    def set_name(self, name: str):
        self.image['name'] = name

    def __len__(self):
        return len(self.image['objects'])

    def __iter__(self):
        return ImageDatasetObjectManipulatorIterator(self.image)

    def delete(self):
        del self.dataset['images'][self.index_of_image]
        del self.image
        if self.parent_iterator is not None:
            self.parent_iterator.deleted()


class ImageDatasetImageManipulatorIterator:
    def __init__(self, dataset: dict):
        self.dataset = dataset
        self.index = 0

    def __next__(self):
        if self.index >= len(self.dataset['images']):
            raise StopIteration
        modifier = ImageDatasetImageManipulator(self.dataset, self.index, self)
        self.index += 1
        return modifier

    def deleted(self):
        self.index -= 1


class ImageDatasetManipulator:
    def __init__(self, dataset: dict):
        self.dataset = dataset
        self.dataset['filters'] = 'dirty'

    def apply_index_filter(self, indices: list):
        self.dataset['images'] = [self.dataset['images'][index] for index in indices]

    def set_name(self, name: str):
        self.dataset['name'] = name

    def get_category_id_name_map(self):
        return self.dataset['category_id_name_map']

    def has_category_id_name_map(self):
        return 'category_id_name_map' in self.dataset

    def set_category_id_name_map(self, category_id_name_map: dict):
        self.dataset['category_id_name_map'] = category_id_name_map

    def __len__(self):
        return len(self.dataset['images'])

    def __iter__(self):
        return ImageDatasetImageManipulatorIterator(self.dataset)
