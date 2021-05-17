import torch
from torch import nn
import torch.nn.functional as F


class _IoUAwareLoss(nn.Module):
    def __init__(self, iou_fn, eps=1e-6, reduction='mean'):
        super(_IoUAwareLoss, self).__init__()
        assert reduction in (None, 'none', 'mean', 'sum')
        self.iou_fn = iou_fn
        self.eps = eps
        self.reduction = reduction

    def __call__(self, bounding_box_predicted, bounding_box_ground_truth, iou_predicted):
        cious = self.iou_fn(bounding_box_predicted, bounding_box_ground_truth, self.eps)
        loss_iou_aware = F.binary_cross_entropy_with_logits(iou_predicted, cious, reduction=self.reduction)
        return loss_iou_aware


class IoUAwareLoss(_IoUAwareLoss):
    def __init__(self, eps=1e-6, reduction='mean'):
        from models.operator.iou.iou import iou
        super(IoUAwareLoss, self).__init__(iou, eps, reduction)


class GIoUAwareLoss(_IoUAwareLoss):
    def __init__(self, eps=1e-6, reduction='mean'):
        from models.operator.iou.giou import giou
        super(GIoUAwareLoss, self).__init__(giou, eps, reduction)


class DIoUAwareLoss(_IoUAwareLoss):
    def __init__(self, eps=1e-6, reduction='mean'):
        from models.operator.iou.diou import diou
        super(DIoUAwareLoss, self).__init__(diou, eps, reduction)


class CIoUAwareLoss(_IoUAwareLoss):
    def __init__(self, eps=1e-6, reduction='mean'):
        from models.operator.iou.ciou import ciou
        super(CIoUAwareLoss, self).__init__(ciou, eps, reduction)
