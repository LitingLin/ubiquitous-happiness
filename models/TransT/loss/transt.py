import torch
import torch.distributed

from torch import nn
from Miscellaneous.torch.distributed import get_world_size, is_dist_available_and_initialized
from Miscellaneous.torch.distributed.reduce_dict import reduce_dict
from models.loss.iou_loss import GIoULoss, CIoULoss
from data.operator.bbox.spatial.vectorized.torch.cxcywh_to_xyxy import box_cxcywh_to_xyxy


class TransTCriterion(nn.Module):
    def __init__(self, weight_dict, eos_coef, reg_smooth_l1=False, ciou=False):
        super().__init__()
        self.weight_dict = weight_dict
        empty_weight = torch.ones((2))
        empty_weight[-1] = eos_coef
        self.register_buffer('empty_weight', empty_weight)
        if ciou:
            self.iou_loss = CIoULoss(reduction='sum')
        else:
            self.iou_loss = GIoULoss(reduction='sum')
        if reg_smooth_l1:
            self.bbox_loss = nn.SmoothL1Loss(reduction='sum')
        else:
            self.bbox_loss = nn.L1Loss(reduction='sum')
        self.ce_loss = nn.CrossEntropyLoss(self.empty_weight, reduction='mean')

    def _do_statistic(self, stats: dict):
        loss_dict_reduced = reduce_dict(stats)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * self.weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in self.weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        return loss_value, loss_dict_reduced_scaled, loss_dict_reduced_unscaled

    def _criterion(self, predicted, label):
        predicted_class, predicted_coord = predicted
        num_boxes_pos, target_feat_map_indices_batch_id_vector, target_feat_map_indices, target_class_label_vector, target_bounding_box_label_matrix = label
        if is_dist_available_and_initialized():
            torch.distributed.all_reduce(num_boxes_pos)
        num_boxes_pos = torch.clamp(num_boxes_pos / get_world_size(), min=1).item()

        src_boxes = predicted_coord[(target_feat_map_indices_batch_id_vector, target_feat_map_indices)]

        losses = {'loss_bbox': self.bbox_loss(src_boxes, target_bounding_box_label_matrix) / num_boxes_pos,
                  'loss_iou': self.iou_loss(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_bounding_box_label_matrix)) / num_boxes_pos,
                  'loss_ce': self.ce_loss(predicted_class.transpose(1, 2), target_class_label_vector)}
        return losses

    def forward(self, predicted, label):
        loss_dict = self._criterion(predicted, label)
        losses = sum(loss_dict[k] * self.weight_dict[k] for k in loss_dict.keys() if k in self.weight_dict)
        return (losses, *self._do_statistic(loss_dict))


def build_transt_criterion(train_config: dict):
    loss_parameters = train_config['train']['loss']['parameters']
    if 'ciou' in loss_parameters['weight']:
        use_ciou = True
        iou_loss_weight = loss_parameters['weight']['ciou']
    elif 'giou' in loss_parameters['weight']:
        use_ciou = False
        iou_loss_weight = loss_parameters['weight']['giou']
    else:
        raise RuntimeError('IoU Loss not defined')
    weight_dict = {'loss_ce': loss_parameters['weight']['cross_entropy'], 'loss_bbox': loss_parameters['weight']['bbox'], 'loss_iou': iou_loss_weight}
    reg_smooth_l1 = False
    if 'bbox_smooth_l1' in loss_parameters and loss_parameters['bbox_smooth_l1']:
        reg_smooth_l1 = True
    return TransTCriterion(weight_dict, loss_parameters['eos_coef'], reg_smooth_l1, use_ciou)
