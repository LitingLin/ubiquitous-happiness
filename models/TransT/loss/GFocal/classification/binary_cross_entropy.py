from models.TransT.loss.common.reduction.default import build_loss_reduction_function


def gfocal_cls_data_adaptor(predicted, label, _):
    cls_score, predicted_bbox, bbox_distribution = predicted
    num_boxes_pos, target_feat_map_indices_batch_id_vector, target_feat_map_indices, target_class_label_vector, target_bounding_box_label_matrix = label
    cls_score = cls_score.flatten(2).transpose(1, 2).flatten(1)
    return True, (cls_score, target_class_label_vector)


def build_binary_cross_entropy(loss_parameters, *_):
    from models.TransT.loss.single_class_quality_gfocal.bce import BCELoss
    return BCELoss(False, reduction='none'), gfocal_cls_data_adaptor, build_loss_reduction_function(loss_parameters)
