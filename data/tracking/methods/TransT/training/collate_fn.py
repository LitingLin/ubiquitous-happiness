import torch
import torch.utils.data


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


def transt_collate_fn(data):
    z_image_list = []
    x_image_list = []
    z_context_list = []
    x_context_list = []
    target_feat_map_indices_list = []
    target_class_label_vector_list = []
    target_bounding_box_label_matrix_list = []
    is_positives = torch.empty([len(data)], dtype=torch.bool)
    if torch.utils.data.get_worker_info() is not None:
        is_positives.share_memory_()
    for index, (z_image, x_image, z_context, x_context, is_positive,
        target_feat_map_indices, target_class_label_vector, target_bounding_box_label_matrix) in enumerate(data):
        z_image_list.append(z_image)
        x_image_list.append(x_image)
        if z_context is not None:
            z_context_list.append(z_context)
        if x_context is not None:
            x_context_list.append(x_context)
        is_positives[index] = is_positive
        target_feat_map_indices_list.append(target_feat_map_indices)
        target_class_label_vector_list.append(target_class_label_vector)
        if target_bounding_box_label_matrix is not None:
            target_bounding_box_label_matrix_list.append(target_bounding_box_label_matrix)

    assert len(z_context_list) == len(x_context_list)

    if len(z_context_list) == 0:
        z_image_batch = torch.stack(z_image_list)
        x_image_batch = torch.stack(x_image_list)
        context = None
    else:
        z_image_batch = z_image_list
        x_image_batch = x_image_list
        context = (torch.stack(z_context_list), torch.stack(x_context_list))

    target_feat_map_indices_batch_id_vector, target_feat_map_indices_batch, num_boxes_pos = \
        batch_collate_target_feat_map_indices(target_feat_map_indices_list)
    target_class_label_vector_batch = torch.stack(target_class_label_vector_list)
    if len(target_bounding_box_label_matrix_list) != 0:
        target_bounding_box_label_matrix_batch = torch.cat(target_bounding_box_label_matrix_list, dim=0)
    else:
        target_bounding_box_label_matrix_batch = None

    # is_positives = torch.tensor(is_positives)

    return (z_image_batch, x_image_batch), \
           (num_boxes_pos, target_feat_map_indices_batch_id_vector, target_feat_map_indices_batch,
            target_class_label_vector_batch, target_bounding_box_label_matrix_batch), \
           is_positives, context


def SiamFC_collate_fn(data):
    z_image_list = []
    x_image_list = []
    z_context_list = []
    x_context_list = []
    is_positives = torch.empty([len(data)], dtype=torch.bool)
    labels = []
    if torch.utils.data.get_worker_info() is not None:
        is_positives.share_memory_()
    for index, (z_image, x_image, z_context, x_context, is_positive, label) in enumerate(data):
        z_image_list.append(z_image)
        x_image_list.append(x_image)
        if z_context is not None:
            z_context_list.append(z_context)
        if x_context is not None:
            x_context_list.append(x_context)
        is_positives[index] = is_positive
        labels.append(label)

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

    return (z_image_batch, x_image_batch), labels, is_positives, context

