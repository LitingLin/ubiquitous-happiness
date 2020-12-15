import torch.nn.functional as F


def l1_bbox_loss(src_boxes, target_boxes, num_boxes):
    loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

    return loss_bbox.sum() / num_boxes
