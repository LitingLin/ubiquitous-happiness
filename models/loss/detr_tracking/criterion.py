from torch import nn
from ..giou import giou_loss
from ..l1_bbox import l1_bbox_loss
import Utils.detr_misc as utils


class DETRTrackingLoss(nn.Module):
    def __init__(self, loss_weights: dict):
        super(DETRTrackingLoss, self).__init__()
        self.losses = {}
        if 'bbox' in loss_weights:
            self.losses[l1_bbox_loss.__name__] = (l1_bbox_loss, loss_weights['bbox'])
        if 'giou' in loss_weights:
            self.losses[giou_loss.__name__] = (giou_loss, loss_weights['giou'])

    def _do_statistic(self, stats: dict):
        stats_reduced = utils.reduce_dict(stats)
        stats_reduced = {k: v.cpu() for k, v in stats_reduced.items()}

        stats_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in stats_reduced.items()}

        stats_reduced_scaled = {k: v * self.losses[k][1]
                                    for k, v in stats_reduced.items()}
        return stats_reduced_unscaled, stats_reduced_scaled

    def forward(self, src_boxes, target_boxes):
        num_boxes = src_boxes.shape[0]
        # expects [num_boxes, 4]
        stats = {}
        losses = []
        for func_name, (func, weight) in self.losses.items():
            loss = func(src_boxes, target_boxes, num_boxes)
            stats[func_name] = loss
            losses.append(loss * weight)

        return (sum(losses), *self._do_statistic(stats))
