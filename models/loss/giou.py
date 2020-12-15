import torch
from Utils import boxes_ops


def giou_loss(src_boxes, target_boxes, num_boxes):
    loss_giou = 1 - torch.diag(boxes_ops.generalized_box_iou(
        boxes_ops.box_cxcywh_to_xyxy(src_boxes),
        boxes_ops.box_cxcywh_to_xyxy(target_boxes)))
    return loss_giou.sum() / num_boxes
