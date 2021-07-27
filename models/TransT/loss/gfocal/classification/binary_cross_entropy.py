
def gfocal_cls_data_adaptor(predicted, label):
    cls_score, predicted_bbox, bbox_distribution = predicted
    num_boxes_pos, target_feat_map_indices_batch_id_vector, target_feat_map_indices, target_class_label_vector, target_bounding_box_label_matrix = label
    return cls_score, target_class_label_vector


def build_binary_cross_entropy(_):
    from models.TransT.loss.single_class_quality_gfocal.bce import BCELoss
    return BCELoss()