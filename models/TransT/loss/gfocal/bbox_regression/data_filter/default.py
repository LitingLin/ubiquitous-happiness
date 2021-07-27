

def default_sample_filter(predicted, label):
    '''
        cls_score: (N, C, H, W)
        predicted_bbox: (N, H, W, 4)
        bbox_distribution: (N, H, W, 4 * (reg_max + 1))
    '''
    cls_score, predicted_bbox, bbox_distribution = predicted

    num_boxes_pos, target_feat_map_indices_batch_id_vector,\
        target_feat_map_indices, target_class_label_vector, target_bounding_box_label_matrix = label
    reduce_mean_(num_boxes_pos)
    num_boxes_pos = max(num_boxes_pos.item(), 1e-4)
    if target_feat_map_indices_batch_id_vector is None:
        return (predicted_bbox, bbox_distribution), None
    else:
        predicted_bbox = predicted_bbox[target_feat_map_indices_batch_id_vector, target_feat_map_indices]
        bbox_distribution = bbox_distribution[target_feat_map_indices_batch_id_vector, target_feat_map_indices]
        return (cls_score, predicted_bbox, bbox_distribution), (num_boxes_pos, target_feat_map_indices_batch_id_vector,
                                                                target_feat_map_indices,
                                                                target_bounding_box_label_matrix)


def build_sample_filter(network_config: dict):
    return default_sample_filter
