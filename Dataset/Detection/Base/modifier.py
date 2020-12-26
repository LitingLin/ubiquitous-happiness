from .dataset import DetectionDataset
import pathlib
from ._impl import _set_image_path
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

    def setCategoryId(self, id_: int):
        assert isinstance(id_, int)
        assert self.context.has_object_category
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

    def sortByImageRatio(self):
        image_sizes = []
        for image in self:
            image_sizes.append(image.getImageSize())
        image_sizes = np.array(image_sizes)
        ratio = image_sizes[:, 1] / image_sizes[:, 0]
        indices = ratio.argsort()
        images = self.dataset.images
        self.dataset.images = [images[index] for index in indices]

    def applyIndicesFilter(self, indices):
        self.dataset.images = [self.dataset.images[index] for index in indices]

    def __iter__(self):
        return DetectionDatasetModifier_ImagesIterator(self.dataset, self.context)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
        return DetectionDatasetImageModifier(self.dataset, index, self.context)
