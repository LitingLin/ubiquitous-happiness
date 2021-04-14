import torch

from torch import nn
from Utils.detr_misc import is_dist_avail_and_initialized, get_world_size, reduce_dict
import Utils.boxes_ops as box_ops
import torch.nn.functional as F


class TransTCriterion(nn.Module):
    def __init__(self, weight_dict, eos_coef):
        super().__init__()
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        empty_weight = torch.ones((2))
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

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
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes_pos)
        num_boxes_pos = torch.clamp(num_boxes_pos / get_world_size(), min=1).item()

        src_boxes = predicted_coord[(target_feat_map_indices_batch_id_vector, target_feat_map_indices)]
        loss_bbox = F.l1_loss(src_boxes, target_bounding_box_label_matrix, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes_pos
        giou, iou = box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_bounding_box_label_matrix))
        giou = torch.diag(giou)
        iou = torch.diag(iou)
        loss_giou = 1 - giou
        iou = iou
        losses['loss_giou'] = loss_giou.sum() / num_boxes_pos
        losses['iou'] = iou.sum() / num_boxes_pos

        loss_ce = F.cross_entropy(predicted_class.transpose(1, 2), target_class_label_vector, self.empty_weight)
        losses['loss_ce'] = loss_ce
        return losses

    def forward(self, predicted, label):
        loss_dict = self._criterion(predicted, label)
        losses = sum(loss_dict[k] * self.weight_dict[k] for k in loss_dict.keys() if k in self.weight_dict)
        return (losses, *self._do_statistic(loss_dict))


def build_transt_criterion(train_config: dict):
    loss_parameters = train_config['train']['loss']
    weight_dict = {'loss_ce': loss_parameters['weight']['cross_entropy'], 'loss_bbox': loss_parameters['weight']['bbox'], 'loss_giou': loss_parameters['weight']['giou']}
    return TransTCriterion(weight_dict, loss_parameters['eos_coef'])
