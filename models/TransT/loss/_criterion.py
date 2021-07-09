import torch.nn as nn
from miscellanies.torch.distributed import is_dist_available_and_initialized, get_world_size
import torch.distributed


def criterion(predicted, label, loss_fn_dict: nn.ModuleDict, loss_target_dict: dict):
    '''
        predicted:
            C: num_classes, L: size_of_feat_map, typically H*W

            cls: (N, C, L) or (N, L)
            reg: (N, L, *)
            qa: same as reg
    '''
    num_boxes_pos, target_feat_map_indices_batch_id_vector, target_feat_map_indices, target_class_label_vector, target_bounding_box_label_matrix = label
    if is_dist_available_and_initialized():
        torch.distributed.all_reduce(num_boxes_pos)

    num_boxes_pos = (num_boxes_pos / get_world_size()).item()

    is_non_positives = target_feat_map_indices_batch_id_vector is None

    predicted_cls = None  # = predicted[0]
    predicted_reg = None  # (P, *) P: num_positives
    predicted_qa = None  # (P, *)
    reg_zero = None
    qa_zero = None

    losses = {}

    for loss_name, loss_fn in loss_fn_dict.items():
        loss_fn_target = loss_target_dict[loss_name]
        if loss_fn_target == 0 and predicted_cls is None:
            predicted_cls = predicted[0]
        elif loss_fn_target == 1 and predicted_reg is None:
            predicted_reg = predicted[1]
            if is_non_positives and reg_zero is None:
                reg_zero = torch.mean(predicted_reg * 0)
            else:
                predicted_reg = predicted_reg[(target_feat_map_indices_batch_id_vector, target_feat_map_indices)]
        elif loss_fn_target == 2 and predicted_qa is None:
            predicted_qa = predicted[3]
            if is_non_positives and qa_zero is None:
                qa_zero = torch.mean(predicted_qa * 0)
            else:
                predicted_qa = predicted_qa[(target_feat_map_indices_batch_id_vector, target_feat_map_indices)]

        if loss_fn_target == 0:
            losses[loss_name] = loss_fn(predicted_cls, target_class_label_vector)
        elif loss_fn_target == 1:
            if is_non_positives:
                losses[loss_name] = reg_zero
            else:
                losses[loss_name] = loss_fn(predicted_reg, target_bounding_box_label_matrix) / num_boxes_pos
        elif loss_fn_target == 2:
            if is_non_positives:
                losses[loss_name] = qa_zero
            else:
                losses[loss_name] = loss_fn(predicted_qa, target_bounding_box_label_matrix) / num_boxes_pos
    return losses
