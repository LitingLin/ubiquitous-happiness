import torch.nn as nn
import torch.nn.functional as F


def bce_loss(pred, target, use_sigmoid=True):
    r"""Quality Focal Loss (QFL) is from `Generalized Focal Loss: Learning
    Qualified and Distributed Bounding Boxes for Dense Object Detection
    <https://arxiv.org/abs/2006.04388>`_.

    Args:
        pred (torch.Tensor): Predicted joint representation of classification
            and quality (IoU) estimation with shape (N, C), C is the number of
            classes.
        target (tuple([torch.Tensor])): Target category label with shape (N,)
            and target quality label with shape (N,).
        beta (float): The beta parameter for calculating the modulating factor.
            Defaults to 2.0.

    Returns:
        torch.Tensor: Loss tensor with shape (N,).
    """
    assert len(target) == 3, """target for QFL must be a tuple of two elements,
        including category label and quality label, respectively"""
    # label denotes the category id, score denotes the quality score
    # N, L
    if use_sigmoid:
        func = F.binary_cross_entropy_with_logits
    else:
        func = F.binary_cross_entropy

    # negatives are supervised by 0 quality score
    pred_sigmoid = pred.sigmoid() if use_sigmoid else pred

    loss = func(pred_sigmoid, target, reduction='none')
    return loss.flatten()


class BCELoss(nn.Module):
    r"""Quality Focal Loss (QFL) is a variant of `Generalized Focal Loss:
    Learning Qualified and Distributed Bounding Boxes for Dense Object
    Detection <https://arxiv.org/abs/2006.04388>`_.

    Args:
        use_sigmoid (bool): Whether sigmoid operation is conducted in QFL.
            Defaults to True.
        beta (float): The beta parameter for calculating the modulating factor.
            Defaults to 2.0.
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Loss weight of current loss.
    """

    def __init__(self,
                 use_sigmoid=True,
                 reduction='mean'):
        super(BCELoss, self).__init__()
        # assert use_sigmoid is True, 'Only sigmoid in QFL supported now.'
        self.use_sigmoid = use_sigmoid
        self.reduction = reduction

    def forward(self,
                pred,
                target):
        """Forward function.

        Args:
            pred (torch.Tensor): Predicted joint representation of
                classification and quality (IoU) estimation with shape (N, C),
                C is the number of classes.
            target (tuple([torch.Tensor])): Target category label with shape
                (N,) and target quality label with shape (N,).
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        cls_loss = bce_loss(
            pred,
            target,
            use_sigmoid=self.use_sigmoid)
        if self.reduction == 'sum':
            return cls_loss.sum()
        elif self.reduction == 'mean':
            return cls_loss.mean()
        else:
            return cls_loss
