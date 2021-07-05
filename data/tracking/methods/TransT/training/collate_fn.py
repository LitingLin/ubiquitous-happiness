import torch
import torch.utils.data
from torch.utils.data.dataloader import default_collate


def batch_collate_target_feat_map_indices(target_feat_map_indices_list):
    batch_ids = []
    batch_target_feat_map_indices = []
    num_boxes_pos = 0
    for index, target_feat_map_indices in enumerate(target_feat_map_indices_list):
        if target_feat_map_indices is None:
            continue
        batch_ids.extend([index for _ in range(len(target_feat_map_indices))])
        batch_target_feat_map_indices.append(target_feat_map_indices)
        num_boxes_pos += len(target_feat_map_indices)

    num_boxes_pos = torch.as_tensor([num_boxes_pos], dtype=torch.float)
    if len(batch_ids) != 0:
        return torch.tensor(batch_ids, dtype=torch.long), torch.cat(batch_target_feat_map_indices), num_boxes_pos
    else:
        return None, None, num_boxes_pos


def collate_different_size_images(images: list):
    collated = []
    shape_collated = {}

    index_of_collated_tensor_list = []
    index_in_collated_tensor_list = []

    for image in images:
        c, h, w = image.shape
        assert c == 3
        if (h, w) not in shape_collated:
            index_of_collated_tensor = len(shape_collated)
            shape_collated[(h, w)] = index_of_collated_tensor
            current_shape_image_list = []
            collated.append(current_shape_image_list)
        else:
            index_of_collated_tensor = shape_collated[(h, w)]
            current_shape_image_list = collated[index_of_collated_tensor]

        index_in_collated_tensor = len(current_shape_image_list)
        current_shape_image_list.append(image)
        index_of_collated_tensor_list.append(index_of_collated_tensor)
        index_in_collated_tensor_list.append(index_in_collated_tensor)

    collated = [torch.stack(image) for image in collated]
    index_of_collated_tensor_list = torch.tensor(index_of_collated_tensor_list)
    index_in_collated_tensor_list = torch.tensor(index_in_collated_tensor_list)
    return collated, index_of_collated_tensor_list, index_in_collated_tensor_list


def transt_collate_fn(data):
    z_image_list = []
    x_image_list = []
    z_context_list = []
    x_context_list = []
    target_feat_map_indices_list = []
    target_class_label_vector_list = []
    target_bounding_box_label_matrix_list = []
    miscellanies_host = []
    miscellanies_element = []

    for index, (z_image, x_image, z_context, x_context, miscellany, miscellany_element,
        target_feat_map_indices, target_class_label_vector, target_bounding_box_label_matrix) in enumerate(data):
        z_image_list.append(z_image)
        x_image_list.append(x_image)
        if z_context is not None:
            z_context_list.append(z_context)
        if x_context is not None:
            x_context_list.append(x_context)
        miscellanies_host.append(miscellany)
        target_feat_map_indices_list.append(target_feat_map_indices)
        target_class_label_vector_list.append(target_class_label_vector)
        if target_bounding_box_label_matrix is not None:
            target_bounding_box_label_matrix_list.append(target_bounding_box_label_matrix)
        if miscellany_element is not None:
            miscellanies_element.append(miscellany_element)

    assert len(z_context_list) == len(x_context_list)

    miscellanies_host = default_collate(miscellanies_host)

    if len(z_context_list) == 0:
        z_image_batch = torch.stack(z_image_list)
        x_image_batch = torch.stack(x_image_list)
        miscellanies_device = None
    else:
        z_image_batch, z_index_of_collated_tensors, z_index_in_collated_tensors = collate_different_size_images(z_image_list)
        x_image_batch, x_index_of_collated_tensors, x_index_in_collated_tensors = collate_different_size_images(
            x_image_list)

        miscellanies_host['z_index_of_collated_tensors'] = z_index_of_collated_tensors
        miscellanies_host['x_index_of_collated_tensors'] = x_index_of_collated_tensors

        miscellanies_device = (torch.stack(z_context_list), torch.stack(x_context_list), z_index_in_collated_tensors, x_index_in_collated_tensors)

    target_feat_map_indices_batch_id_vector, target_feat_map_indices_batch, num_boxes_pos = \
        batch_collate_target_feat_map_indices(target_feat_map_indices_list)
    target_class_label_vector_batch = torch.stack(target_class_label_vector_list)
    if len(target_bounding_box_label_matrix_list) != 0:
        target_bounding_box_label_matrix_batch = torch.cat(target_bounding_box_label_matrix_list, dim=0)
    else:
        target_bounding_box_label_matrix_batch = None

    if len(miscellanies_element) == 0:
        miscellanies_element = None

    return (z_image_batch, x_image_batch), \
           (num_boxes_pos, target_feat_map_indices_batch_id_vector, target_feat_map_indices_batch,
            target_class_label_vector_batch, target_bounding_box_label_matrix_batch), \
           miscellanies_host, miscellanies_device, miscellanies_element


def SiamFC_collate_fn(data):
    z_image_list = []
    x_image_list = []
    z_context_list = []
    x_context_list = []
    labels = []
    miscellanies = []

    collate_miscellanies = None

    for index, (z_image, x_image, z_context, x_context, miscellany, collate_miscellany, label) in enumerate(data):
        z_image_list.append(z_image)
        x_image_list.append(x_image)
        if z_context is not None:
            z_context_list.append(z_context)
        if x_context is not None:
            x_context_list.append(x_context)
        labels.append(label)
        miscellanies.append(miscellany)
        collate_miscellanies = collate_miscellany

    assert len(z_context_list) == len(x_context_list)

    if len(z_context_list) == 0:
        z_image_batch = torch.stack(z_image_list)
        x_image_batch = torch.stack(x_image_list)
        context = None
    else:
        z_image_batch = z_image_list
        x_image_batch = x_image_list
        context = (torch.stack(z_context_list), torch.stack(x_context_list))

    labels = torch.stack(labels)

    if collate_miscellanies:
        miscellanies = default_collate(miscellanies)

    return (z_image_batch, x_image_batch), labels, miscellanies, context
