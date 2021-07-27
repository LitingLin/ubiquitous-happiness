import torch


def reduction_by_classification_score(loss, predicted, label, context: dict):
    classification_score = predicted[0]

    num_boxes_pos, target_feat_map_indices_batch_id_vector,\
        target_feat_map_indices, target_bounding_box_label_matrix = label
    target_weights = context['target_weights']
    loss * target_weights

def build_loss_reduction_function(_):
    pass