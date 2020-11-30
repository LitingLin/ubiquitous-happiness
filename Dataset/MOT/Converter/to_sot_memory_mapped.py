from Dataset.MOT.Base.dataset import MultipleObjectTrackingDataset
from Dataset.SOT.Base.MemoryMapped.dataset import SingleObjectTrackingDataset_MemoryMapped
from Dataset.Types.MemoryMapped.string_array import StringArrayMemoryMappedConstructor
from Dataset.Types.MemoryMapped.digit_array import DigitMatrixMemoryMappedConstructor
from Dataset.CacheService.common import _getCachePath
import os
import numpy as np


def to_sot_memory_mapped(dataset: MultipleObjectTrackingDataset, sot_dataset: SingleObjectTrackingDataset_MemoryMapped):
    sot_dataset.root_path = str(sot_dataset.root_path)
    sot_dataset.category_name_id_mapper = dataset.category_name_id_mapper
    sot_dataset.category_names = dataset.category_names

    cache_path, cache_file_prefix = _getCachePath(sot_dataset)
    if not os.path.exists(cache_path):
        os.mkdir(cache_path)
    image_paths_memory_mapped_file_name = os.path.join(cache_path, cache_file_prefix + '-image_paths.numpy')
    bounding_boxes_memory_mapped_file_name = os.path.join(cache_path, cache_file_prefix + '-bounding_boxes.numpy')
    image_paths = StringArrayMemoryMappedConstructor(image_paths_memory_mapped_file_name)
    bounding_boxes = DigitMatrixMemoryMappedConstructor(bounding_boxes_memory_mapped_file_name)
    sequence_attributes_indices = []
    sequence_category_ids = []
    current_index = 0
    sequence_has_attribute_fps = dataset.hasAttibuteFPS()
    if sequence_has_attribute_fps:
        sequence_fps_s = []

    for sequence in dataset:
        for track in sequence.getTrackIterator():
            sequence_attributes_indices.append(current_index)
            for frame in track:
                image_path = str(frame.frame.image_path)
                is_present = frame.getAttributeIsPresent()
                if is_present is False:
                    bounding_box = [-1, -1, -1, -1]
                else:
                    bounding_box = frame.getBoundingBox()
                    if bounding_box[0] + bounding_box[2] <= 0 or bounding_box[1] + bounding_box[3] <= 0:
                        bounding_box = [-1, -1, -1, -1]
                image_paths.add(image_path)
                bounding_boxes.add(bounding_box)
                current_index += 1
            if current_index == sequence_attributes_indices[-1]:
                sequence_attributes_indices.pop()
                continue
            sequence_attributes_indices.append(current_index)
            category_id = track.getObject().getCategoryId()
            sequence_category_ids.append(category_id)
            if sequence_has_attribute_fps:
                sequence_fps_s.append(sequence.getFPS())

    sot_dataset.image_paths = image_paths.construct()
    sot_dataset.bounding_boxes = bounding_boxes.construct()
    sot_dataset.sequence_attributes_indices = np.array(sequence_attributes_indices)
    sot_dataset.sequence_category_ids = np.array(sequence_category_ids)
    if sequence_has_attribute_fps:
        sot_dataset.sequence_fps_s = np.array(sequence_fps_s)
