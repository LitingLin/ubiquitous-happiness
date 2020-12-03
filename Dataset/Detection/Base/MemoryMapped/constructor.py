from Dataset.Detection.Base.dataset import DetectionDataset
from Dataset.Detection.Base.MemoryMapped.dataset import DetectionDataset_MemoryMapped
import numpy as np
from Dataset.CacheService.constructor import DatasetConstructor_CacheService_Base
from Dataset.Types.MemoryMapped.string_array import StringArrayMemoryMappedConstructor
from Dataset.Types.MemoryMapped.digit_array import DigitMatrixMemoryMappedConstructor
from Dataset.CacheService.common import _getCachePath
import os


class DetectionDatasetConstructor_MemoryMapped(DatasetConstructor_CacheService_Base):
    def __init__(self, dataset: DetectionDataset_MemoryMapped):
        super(DetectionDatasetConstructor_MemoryMapped, self).__init__(dataset)

    def loadFrom(self, dataset: DetectionDataset):
        has_attribute_category = dataset.hasAttributeCategory()
        has_attribute_is_presents = dataset.hasAttibuteIsPresent()
        self.dataset.name = dataset.name
        self.dataset.data_version = dataset.data_version
        self.dataset.data_split = dataset.data_split
        self.dataset.filters = dataset.filters
        if self.root_path is None:
            self.dataset.root_path = dataset.root_path
        self.dataset.root_path = str(self.dataset.root_path)
        if has_attribute_category:
            self.dataset.category_id_name_mapper = dataset.category_id_name_mapper

        cache_path, cache_file_prefix = _getCachePath(self.dataset)
        if not os.path.exists(cache_path):
            os.mkdir(cache_path)

        image_paths_memory_mapped_file_name = os.path.join(cache_path, cache_file_prefix + '-image_paths.numpy')
        bounding_boxes_memory_mapped_file_name = os.path.join(cache_path, cache_file_prefix + '-bounding_boxes.numpy')

        image_paths_constructor = StringArrayMemoryMappedConstructor(image_paths_memory_mapped_file_name)
        bounding_boxes_constructor = DigitMatrixMemoryMappedConstructor(bounding_boxes_memory_mapped_file_name)

        category_ids = []
        image_sizes = []
        image_attributes_indices = []
        is_presents = []

        current_index = 0
        for image in dataset.images:
            image_attributes_indices.append(current_index)
            image_paths_constructor.add(str(image.image_path))
            image_sizes.append(image.size)
            for object_ in image.objects:
                bounding_boxes_constructor.add(object_.bounding_box)
                if has_attribute_category:
                    category_ids.append(object_.category_id)
                if has_attribute_is_presents:
                    is_presents.append(object_.is_present)
                current_index += 1
        self.dataset.image_paths = image_paths_constructor.construct()
        self.dataset.bounding_boxes = bounding_boxes_constructor.construct()
        if has_attribute_category:
            self.dataset.category_ids = np.array(category_ids)
        self.dataset.image_attributes_indices = np.array(image_attributes_indices)
        self.dataset.image_sizes = np.array(image_sizes)
        if has_attribute_is_presents:
            self.dataset.is_presents = np.array(is_presents)

    def setDatasetAttribute(self, name, value):
        self.dataset.attributes[name] = value
