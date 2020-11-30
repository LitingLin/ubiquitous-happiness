from .dataset import DetectionDataset
import pathlib
from ._impl import _set_image_path, _get_or_allocate_category_id
from typing import Dict
import numpy as np

# cache python objects
class _Context:
    root_path: pathlib.Path
    has_object_category: bool
    has_is_present: bool


class DetectionDatasetImageObjectModifier:
    def __init__(self, dataset, image, index: int, context, parent_iterator=None):
        self.dataset = dataset
        self.image = image
        self.index = index
        self.object_ = image.objects[index]
        self.context = context
        self.parent_iterator = parent_iterator

    def setBoundingBox(self, bounding_box):
        self.object_.bounding_box = bounding_box

    def getBoundingBox(self):
        return self.object_.bounding_box

    def setIsPresent(self, is_present):
        assert isinstance(is_present, bool)
        assert self.context.has_is_present
        self.object_.is_present = is_present

    def setCategoryName(self, name: str):
        assert isinstance(name, str)
        assert self.context.has_object_category
        id_ = _get_or_allocate_category_id(name, self.dataset.category_names, self.dataset.category_name_id_mapper)
        self.object_.category_id = id_

    def delete(self):
        self.image.objects.pop(self.index)
        if self.parent_iterator is not None:
            self.parent_iterator.index -= 1


class DetectionDatasetModifier_ImageObjectsIterator:
    def __init__(self, dataset, image, context):
        self.dataset = dataset
        self.image = image
        self.index = 0
        self.context = context

    def __next__(self):
        if self.index >= len(self.image.objects):
            raise StopIteration
        image_object = DetectionDatasetImageObjectModifier(self.dataset, self.image, self.index, self.context, self)
        self.index += 1
        return image_object


class DetectionDatasetImageModifier:
    def __init__(self, dataset: DetectionDataset, index: int, context, parent_iterator=None):
        self.dataset = dataset
        self.context = context
        self.index = index
        self.image = dataset.images[index]
        self.parent_iterator = parent_iterator

    def setImageName(self, name: str):
        self.image.name = name

    def getImageSize(self):
        return self.image.size

    def setImagePath(self, path: str):
        self.image.size, self.image.image_path = _set_image_path(self.context.root_path, path)

    def __len__(self):
        return len(self.image.objects)

    def __iter__(self):
        return DetectionDatasetModifier_ImageObjectsIterator(self.dataset, self.image, self.context)

    def __getitem__(self, index: int):
        return DetectionDatasetImageObjectModifier(self.dataset, self.image, self.index, self.context)

    def delete(self):
        self.dataset.images.pop(self.index)
        if self.parent_iterator is not None:
            self.parent_iterator.index -= 1


class DetectionDatasetModifier_ImagesIterator:
    def __init__(self, dataset: DetectionDataset, context):
        self.dataset = dataset
        self.index = 0
        self.context = context

    def __next__(self):
        if self.index >= len(self.dataset):
            raise StopIteration
        image_modifier = DetectionDatasetImageModifier(self.dataset, self.index, self.context, self)
        self.index += 1
        return image_modifier


class DetectionDatasetModifier:
    def __init__(self, dataset: DetectionDataset):
        self.dataset = dataset
        self.context = _Context()
        self.context.root_path = pathlib.Path(dataset.root_path)
        self.context.has_object_category = dataset.hasAttributeCategory()
        self.context.has_is_present = dataset.hasAttibuteIsPresent()

    def setRootPath(self, root_path: str):
        self.context.root_path = pathlib.Path(root_path)
        self.dataset.root_path = root_path

    def setDatasetName(self, name: str):
        assert isinstance(name, str)
        self.dataset.name = name

    def shrinkCategoryNames(self):
        assert self.context.has_object_category

        category_ids = set()

        for image in self.dataset.images:
            for object_ in image.objects:
                category_ids.add(object_.category_id)

        if len(category_ids) == len(self.dataset.category_names):
            return
        category_ids = sorted(category_ids)
        old_new_category_id_mapper = {old: new for new, old in enumerate(category_ids)}
        new_category_names = [self.dataset.category_names[old_category_id] for old_category_id in category_ids]
        new_category_name_id_mapper = {name: id_ for id_, name in enumerate(new_category_names)}
        self.dataset.category_names = new_category_names
        self.dataset.category_name_id_mapper = new_category_name_id_mapper
        for image in self.dataset.images:
            for object_ in image.objects:
                object_.category_id = old_new_category_id_mapper[object_.category_id]

    def removeAbsentObjects(self):
        if not self.context.has_is_present:
            return
        for image in self:
            for object_ in image:
                if object_.object_.is_present is False:
                    object_.delete()
        self.dataset.attributes['has_is_present_attr'] = False

    def removeZeroAnnoationImages(self):
        for image in self:
            if len(image) == 0:
                image.delete()

    def removeZeroAnnotationObjects(self):
        for image in self:
            for object_ in image:
                if object_.object_.bounding_box is None:
                    object_.delete()

    def applyCategoryMapping(self, map: Dict[str, str]):
        for old_name, new_name in map.items():
            if old_name == new_name:
                continue
            if new_name not in self.dataset.category_name_id_mapper:
                id_ = self.dataset.category_name_id_mapper[old_name]
                self.dataset.category_name_id_mapper.pop(old_name)
                self.dataset.category_name_id_mapper[new_name] = id_
                self.dataset.category_names[id_] = new_name
            else:
                old_id = self.dataset.category_name_id_mapper[old_name]
                new_id = self.dataset.category_name_id_mapper[new_name]

                for image in self.dataset.images:
                    for object_ in image.objects:
                        if object_.category_id == old_id:
                            object_.category_id = new_id

    def filterCategories(self, includes=None, excludes=None):
        assert self.context.has_object_category

        if includes is not None:
            includes = [self.dataset.category_name_id_mapper[name] for name in includes]
            includes = set(includes)
            for image in self:
                for object_ in image:
                    if object_.object_.category_id not in includes:
                        object_.delete()
        if excludes is not None:
            excludes = [self.dataset.category_name_id_mapper[name] for name in excludes]
            excludes = set(excludes)
            for image in self:
                for object_ in image:
                    if object_.object_.category_id in excludes:
                        object_.delete()

    def sortByImageRatio(self):
        image_sizes = []
        for image in self:
            image_sizes.append(image.getImageSize())
        image_sizes = np.array(image_sizes)
        ratio = image_sizes[:, 1] / image_sizes[:, 0]
        indices = ratio.argsort()
        images = self.dataset.images
        self.dataset.images = [images[index] for index in indices]

    def __iter__(self):
        return DetectionDatasetModifier_ImagesIterator(self.dataset, self.context)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
        return DetectionDatasetImageModifier(self.dataset, index, self.context)
