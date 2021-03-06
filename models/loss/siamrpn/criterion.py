from torch import nn
import torch.nn.functional as F
from models.loss.siamrpn.func import select_cross_entropy_loss, weight_l1_loss


def _log_softmax(cls):
    b, a2, h, w = cls.size()
    cls = cls.view(b, 2, a2 // 2, h, w)
    cls = cls.permute(0, 2, 3, 4, 1).contiguous()
    cls = F.log_softmax(cls, dim=4)
    return cls


class SiamRPNLoss(nn.Module):
    def __init__(self, cls_weight, loc_weight):
        super(SiamRPNLoss, self).__init__()
        self.cls_weight = cls_weight
        self.loc_weight = loc_weight

    def forward(self, output, target):
        cls, loc = output
        label_cls, label_loc, label_loc_weight = target
        # get loss
        cls = _log_softmax(cls)
        cls_loss = select_cross_entropy_loss(cls, label_cls)
        loc_loss = weight_l1_loss(loc, label_loc, label_loc_weight)

        outputs = {'total_loss': self.cls_weight * cls_loss + self.loc_weight * loc_loss, 'cls_loss': cls_loss,
                   'loc_loss': loc_loss}

        return outputs
