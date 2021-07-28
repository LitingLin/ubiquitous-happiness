def positive_only_generate_weight_by_classification_score_sample_filter(predicted, label, context):
    '''
        cls_score: (N, C, H, W)
        predicted_bbox: (N, H, W, 4)
        bbox_distribution: (N, H, W, 4 * (reg_max + 1))
    '''

    num_boxes_pos, target_feat_map_indices_batch_id_vector,\
        target_feat_map_indices, target_class_label_vector, target_bounding_box_label_matrix = label
    if target_feat_map_indices_batch_id_vector is None:
        return predicted, None
    else:
        cls_score, predicted_bbox, bbox_distribution = predicted
        N, H, W, _ = predicted_bbox.shape
        predicted_bbox = predicted_bbox.view(N, H * W, 4)
        assert bbox_distribution.shape[0] == N and bbox_distribution.shape[1] == H and bbox_distribution.shape[2] == W
        bbox_distribution = bbox_distribution.view(N, H * W, -1)

        weight_targets = cls_score.detach().flatten(2).transpose(1, 2).flatten(1)
        weight_targets = weight_targets[target_feat_map_indices_batch_id_vector, target_feat_map_indices].flatten()
        context['sample_weight'] = weight_targets
        predicted_bbox = predicted_bbox[target_feat_map_indices_batch_id_vector, target_feat_map_indices]
        bbox_distribution = bbox_distribution[target_feat_map_indices_batch_id_vector, target_feat_map_indices]
        return (cls_score, predicted_bbox, bbox_distribution), (num_boxes_pos, target_bounding_box_label_matrix)


def build_data_filter(_):
    return positive_only_generate_weight_by_classification_score_sample_filter
