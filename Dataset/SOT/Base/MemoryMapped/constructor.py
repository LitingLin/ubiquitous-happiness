from Dataset.SOT.Base.MemoryMapped.dataset import SingleObjectTrackingDataset_MemoryMapped
from Dataset.SOT.Base.dataset import SingleObjectTrackingDataset
import numpy as np
from Dataset.CacheService.constructor import DatasetConstructor_CacheService_Base
from Utils.get_tight_bounding_box import get_tight_rectangular_bounding_box
from Dataset.Types.MemoryMapped.string_array import StringArrayMemoryMappedConstructor
from Dataset.Types.MemoryMapped.digit_array import DigitMatrixMemoryMappedConstructor
from Dataset.CacheService.common import _getCachePath
import os


class SingleObjectTrackingDatasetConstructor_MemoryMapped(DatasetConstructor_CacheService_Base):
    dataset: SingleObjectTrackingDataset_MemoryMapped
    def __init__(self, dataset: SingleObjectTrackingDataset_MemoryMapped):
        super(SingleObjectTrackingDatasetConstructor_MemoryMapped, self).__init__(dataset)

    def loadFrom(self, dataset: SingleObjectTrackingDataset):
        sequences_has_object_category_attribute = dataset.attributes['has_object_category_attr']
        sequences_has_fps_attribute = dataset.attributes['has_fps_attr']
        if sequences_has_object_category_attribute:
            self.dataset.category_id_name_mapper = dataset.category_id_name_mapper
        self.dataset.filters = dataset.filters
        self.dataset.data_type = dataset.data_split
        self.dataset.data_version = dataset.data_version
        self.dataset.name = dataset.name

        if self.root_path is None:
            self.dataset.root_path = dataset.root_path
        dataset.root_path = str(dataset.root_path)
        cache_path, cache_file_prefix = _getCachePath(self.dataset)
        if not os.path.exists(cache_path):
            os.mkdir(cache_path)
        image_paths_memory_mapped_file_name = os.path.join(cache_path, cache_file_prefix + '-image_paths.numpy')
        bounding_boxes_memory_mapped_file_name = os.path.join(cache_path, cache_file_prefix + '-bounding_boxes.numpy')
        image_paths_constructor = StringArrayMemoryMappedConstructor(image_paths_memory_mapped_file_name)
        bounding_boxes_constructor = DigitMatrixMemoryMappedConstructor(bounding_boxes_memory_mapped_file_name)
        sequence_attributes_indices = [] # with [begin, end) pair
        sequence_category_ids = []
        sequence_fpses = []
        current_index = 0

        sequence_names = []

        for sequence in dataset.sequences:
            sequence_names.append(sequence.name)
            has_present_attr = sequence.attributes['has_present_attr']
            sequence_attributes_indices.append(current_index)
            if has_present_attr:
                for frame in sequence.frames:
                    image_paths_constructor.add(str(frame.image_path))
                    if hasattr(frame, 'bounding_box'):
                        if frame.bounding_box is None or not frame.is_present:
                            bounding_box = [-1, -1, -1, -1]
                        else:
                            bounding_box = frame.bounding_box
                            if len(bounding_box) != 4:
                                bounding_box = get_tight_rectangular_bounding_box(bounding_box)
                    else:
                        bounding_box = [-1, -1, -1, -1]
                    bounding_boxes_constructor.add(bounding_box)
                    current_index += 1
            else:
                for frame in sequence.frames:
                    image_paths_constructor.add(str(frame.image_path))
                    if hasattr(frame, 'bounding_box'):
                        if frame.bounding_box is None:
                            bounding_box = [-1, -1, -1, -1]
                        else:
                            bounding_box = frame.bounding_box
                            if len(bounding_box) != 4:
                                bounding_box = get_tight_rectangular_bounding_box(bounding_box)
                    else:
                        bounding_box = [-1, -1, -1, -1]
                    bounding_boxes_constructor.add(bounding_box)
                    current_index += 1
            if current_index == sequence_attributes_indices[-1]:
                sequence_attributes_indices.pop()
                continue
            sequence_attributes_indices.append(current_index)
            sequence_category_ids.append(sequence.category_id)
            sequence_fpses.append(sequence.fps)
        self.dataset.image_paths = image_paths_constructor.construct()
        self.dataset.bounding_boxes = bounding_boxes_constructor.construct()
        self.dataset.sequence_attributes_indices = np.array(sequence_attributes_indices)
        if sequences_has_object_category_attribute:
            self.dataset.sequence_category_ids = np.array(sequence_category_ids)
        if sequences_has_fps_attribute:
            self.dataset.sequence_fps_s = np.append(sequence_fpses)
        self.dataset.sequence_names = sequence_names

    def setDatasetAttribute(self, name, value):
        self.dataset.attributes[name] = value
