from Dataset.Detection.Base.dataset import DetectionDataset
from Dataset.Detection.Base.image import DetectionDatasetImage
from Dataset.Detection.Base.object import DetectionDatasetObject
from typing import Dict, List
from Dataset.CacheService.constructor import DatasetConstructor_CacheService_Base
from ._impl import _set_image_path


class DetectionDatasetConstructor(DatasetConstructor_CacheService_Base):
    dataset: DetectionDataset
    image: DetectionDatasetImage

    def __init__(self, dataset: DetectionDataset):
        super(DetectionDatasetConstructor, self).__init__(dataset)
        self.has_category_attribute = None
        self.has_is_present_attribute = None

    def setDatasetAttribute(self, name: str, value):
        self.dataset.attributes[name] = value

    def beginInitializeImage(self):
        self.image = DetectionDatasetImage()
        self.image.objects = []
        self.image.attributes = {}

    def endInitializeImage(self):
        assert self.image is not None
        self.dataset.images.append(self.image)
        self.image = None

    def setImageName(self, name: str):
        self.image.name = name

    def setImagePath(self, image_path: str):
        self.image.size, self.image.image_path = _set_image_path(self.root_path, image_path)
        return self.image.size

    def setImageAttribute(self, name, value):
        self.image.attributes[name] = value

    def addImageAttributes(self, attributes):
        self.image.attributes.update(attributes)

    def addObject(self, bounding_box: List, category_name: str=None, is_present: bool=None, attributes: Dict = None):
        assert isinstance(bounding_box, list) or isinstance(bounding_box, tuple)
        if category_name is not None:
            assert isinstance(category_name, str)
            if self.has_category_attribute is None:
                self.has_category_attribute = True
            elif self.has_category_attribute is not True:
                raise ValueError
        else:
            if self.has_category_attribute is None:
                self.has_category_attribute = False
            elif self.has_category_attribute is not False:
                raise ValueError

        if is_present is not None:
            assert isinstance(is_present, bool)
            if self.has_is_present_attribute is None:
                self.has_is_present_attribute = True
            elif self.has_is_present_attribute is not True:
                raise ValueError
        else:
            if self.has_is_present_attribute is None:
                self.has_is_present_attribute = False
            elif self.has_is_present_attribute is not False:
                raise ValueError

        object_ = DetectionDatasetObject()

        object_.bounding_box = bounding_box
        object_.attributes = attributes

        if category_name is not None:
            if category_name not in self.dataset.category_name_id_mapper:
                id_ = len(self.dataset.category_names)
                self.dataset.category_names.append(category_name)
                self.dataset.category_name_id_mapper[category_name] = id_
            else:
                id_ = self.dataset.category_name_id_mapper[category_name]
            object_.category_id = id_
        if is_present is not None:
            object_.is_present = is_present
        self.image.objects.append(object_)

    def performStatistic(self):
        self.dataset.attributes['has_object_category_attr'] = self.has_category_attribute
        self.dataset.attributes['has_is_present_attr'] = self.has_is_present_attribute
        for image in self.dataset:
            if len(image) != 1:
                self.dataset.attributes['single_object'] = False
                return
        self.dataset.attributes['single_object'] = True
