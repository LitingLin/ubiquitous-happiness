from miscellanies.torch.distributed.reduce_mean import reduce_mean_


def default_global_data_filter(predicted, label):
    num_boxes_pos, target_feat_map_indices_batch_id_vector,\
        target_feat_map_indices, target_class_label_vector, target_bounding_box_label_matrix = label

    reduce_mean_(num_boxes_pos)
    num_boxes_pos = max(num_boxes_pos.item(), 1e-4)
    return predicted, (num_boxes_pos, target_feat_map_indices_batch_id_vector,
        target_feat_map_indices, target_class_label_vector, target_bounding_box_label_matrix)


def build_global_data_filter(_):
    return default_global_data_filter
