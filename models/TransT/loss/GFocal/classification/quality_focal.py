import torch
from data.operator.bbox.spatial.vectorized.torch.cxcywh_to_xyxy import box_cxcywh_to_xyxy
from models.TransT.loss.common.reduction.default import build_loss_reduction_function


class QualityFocalDataAdaptor:
    def __init__(self, quality_fn):
        self.quality_fn = quality_fn

    def __call__(self, predicted, label, _):
        cls_score, predicted_bbox, bbox_distribution = predicted
        num_boxes_pos, target_feat_map_indices_batch_id_vector, target_feat_map_indices, target_class_label_vector, target_bounding_box_label_matrix = label

        N, num_classes, H, W = cls_score.shape

        quality_score = torch.zeros((N, H * W), dtype=cls_score.dtype, device=cls_score.device)
        cls_score = cls_score.flatten(2).transpose(1, 2).flatten(1)

        if target_feat_map_indices_batch_id_vector is not None:
            predicted_bbox = predicted_bbox.view(N, H * W, 4)
            predicted_bbox = predicted_bbox[target_feat_map_indices_batch_id_vector, target_feat_map_indices]
            quality_score[
                target_feat_map_indices_batch_id_vector, target_feat_map_indices] = \
                self.quality_fn(box_cxcywh_to_xyxy(predicted_bbox.detach()),
                                box_cxcywh_to_xyxy(target_bounding_box_label_matrix))
        return True, (cls_score, (target_class_label_vector, quality_score, (target_feat_map_indices_batch_id_vector, target_feat_map_indices)))


def build_quality_focal(loss_parameters: dict, *_):
    from models.TransT.loss.single_class_quality_gfocal.quality_gfocal import QualityFocalLoss
    assert loss_parameters['quality_function'] == 'IoU'

    from models.operator.iou.iou import iou

    return QualityFocalLoss(False, loss_parameters['beta'], 'none'), QualityFocalDataAdaptor(iou), build_loss_reduction_function(loss_parameters)
