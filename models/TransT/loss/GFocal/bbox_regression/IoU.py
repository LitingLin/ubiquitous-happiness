from data.operator.bbox.spatial.vectorized.torch.cxcywh_to_xyxy import box_cxcywh_to_xyxy
from models.TransT.loss.common.reduction.default import build_loss_reduction_function


def iou_loss_data_adaptor(pred, label, _):
    (cls_score, predicted_bbox, bbox_distribution) = pred
    if label is None:
        return False, predicted_bbox.sum() * 0
    (num_boxes_pos, target_bounding_box_label_matrix) = label
    return True, (box_cxcywh_to_xyxy(predicted_bbox), box_cxcywh_to_xyxy(target_bounding_box_label_matrix))


def build_IoU(loss_parameters, *_):
    iou_loss_type = loss_parameters['type']
    if iou_loss_type == 'IoU':
        from models.loss.iou_loss import IoULoss
        iou_loss = IoULoss(reduction='none')
    elif iou_loss_type == 'GIoU':
        from models.loss.iou_loss import GIoULoss
        iou_loss = GIoULoss(reduction='none')
    elif iou_loss_type == 'DIoU':
        from models.loss.iou_loss import DIoULoss
        iou_loss = DIoULoss(reduction='none')
    elif iou_loss_type == 'CIoU':
        from models.loss.iou_loss import CIoULoss
        iou_loss = CIoULoss(reduction='none')
    else:
        raise NotImplementedError(f'Unknown value: {iou_loss_type}')
    return iou_loss, iou_loss_data_adaptor, build_loss_reduction_function(loss_parameters)
