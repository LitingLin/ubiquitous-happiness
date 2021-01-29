from Dataset.Base.Engine.memory_mapped import ListMemoryMappedConstructor
from Dataset.Type.bounding_box_format import BoundingBoxFormat
import numpy as np
import Dataset.Base.Common.Operator.bounding_box


def memory_mapped_constructor_common_preliminary_works(base_dataset: dict, base_dataset_type: str, path: str,
                                                       bounding_box_format: BoundingBoxFormat, scheme_version: int,
                                                       target_dataset_type_name: str,
                                                       dataset_key_exclude_list: list):

    assert base_dataset['type'] == base_dataset_type
    if 'filters' in base_dataset:
        assert base_dataset['filters'] != 'dirty'
        base_dataset_filters = base_dataset['filters']
    else:
        base_dataset_filters = []

    constructor = ListMemoryMappedConstructor(path)

    dataset_attributes = {'name': base_dataset['name'], 'split': base_dataset['split'],
                          'version': [scheme_version, base_dataset['version'][1]],
                          'filters': base_dataset_filters, 'type': target_dataset_type_name,
                          'bounding_box_format': bounding_box_format.name}

    if 'category_id_name_map' in base_dataset:
        dataset_attributes['category_id_name_map'] = base_dataset['category_id_name_map']

    for dataset_attribute_key, dataset_attribute_value in base_dataset.items():
        if dataset_attribute_key in dataset_key_exclude_list:
            continue
        dataset_attributes[dataset_attribute_key] = dataset_attribute_value

    constructor.append(dataset_attributes)
    return constructor


def memory_mapped_constructor_commit_data(data_matrix, constructor):
    index_matrix = []
    current_index = 2

    for data_row in data_matrix:
        indices_row = []
        for data in data_row:
            if data is None:
                indices_row.append(-1)
            else:
                indices_row.append(current_index)
                current_index += 1
        index_matrix.append(indices_row)

    index_matrix = np.array(index_matrix)

    constructor.append(index_matrix)

    for data_row in data_matrix:
        for data in data_row:
            if data is not None:
                constructor.append(data)

    return constructor.construct()


def memory_mapped_constructor_get_bounding_box(base_object: dict, image_size, target_bounding_box_format):
    object_bounding_box, object_bounding_box_format, object_bounding_box_validity_flag = Dataset.Base.Common.ops.get_bounding_box(
        base_object)
    if object_bounding_box_validity_flag is None:
        object_bounding_box_validity_flag = Dataset.Base.Common.Operator.bounding_box.check_bounding_box_validity_by_intersection_over_image(
            object_bounding_box, object_bounding_box_format, image_size)
    object_bounding_box = Dataset.Base.Common.Operator.bounding_box.convert_bounding_box_format(object_bounding_box,
                                                                                                object_bounding_box_format,
                                                                                                target_bounding_box_format)
    return object_bounding_box, object_bounding_box_validity_flag


def memory_mapped_constructor_generate_bounding_box_matrix(bounding_box_matrix):
    if all([bounding_box is None for bounding_box in bounding_box_matrix]):
        bounding_box_matrix = None
    else:
        bounding_box_matrix = [[-1, -1, -1, -1] if bounding_box is None else bounding_box for
                                      bounding_box in bounding_box_matrix]
    if bounding_box_matrix is not None:
        bounding_box_matrix = np.array(bounding_box_matrix)
    return bounding_box_matrix


def memory_mapped_constructor_generate_bounding_box_validity_flag_vector(bounding_box_validity_flag_matrix):
    if all([bounding_box_validity_flag is None or bounding_box_validity_flag is True for
            bounding_box_validity_flag in bounding_box_validity_flag_matrix]):
        bounding_box_validity_flag_matrix = None
    else:
        bounding_box_validity_flag_matrix = [
            True if bounding_box_validity_flag is None else bounding_box_validity_flag for
            bounding_box_validity_flag in bounding_box_validity_flag_matrix]
    if bounding_box_validity_flag_matrix is not None:
        bounding_box_validity_flag_matrix = np.array(bounding_box_validity_flag_matrix)
    return bounding_box_validity_flag_matrix
