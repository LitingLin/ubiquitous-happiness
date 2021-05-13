import torch
import torch.distributed

from torch import nn
from Miscellaneous.torch.distributed import get_world_size, is_dist_available_and_initialized
from Miscellaneous.torch.distributed.reduce_dict import reduce_dict
from models.loss.gaussian_focal_loss import GaussianFocalLoss
from models.loss.iou_and_iou_aware_loss import CIoUAndIoUAwareLoss
from data.operator.bbox.spatial.vectorized.torch.cxcywh_to_xyxy import box_cxcywh_to_xyxy


class TransTExp1Criterion(nn.Module):
    def __init__(self, weight_dict, cls_smooth_l1=True):
        super().__init__()
        self.weight_dict = weight_dict
        self.cls_loss = GaussianFocalLoss(reduction='mean')
        self.iou_loss = CIoUAndIoUAwareLoss(reduction='sum')
        if cls_smooth_l1:
            self.bbox_loss = nn.SmoothL1Loss(reduction='sum')
        else:
            self.bbox_loss = nn.L1Loss(reduction='sum')

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
        predicted_class, predicted_coord, predicted_ious = predicted
        num_boxes_pos, target_feat_map_indices_batch_id_vector, target_feat_map_indices, target_class_label_vector, target_bounding_box_label_matrix = label
        if is_dist_available_and_initialized():
            torch.distributed.all_reduce(num_boxes_pos)
        num_boxes_pos = torch.clamp(num_boxes_pos / get_world_size(), min=1).item()

        src_boxes = predicted_coord[(target_feat_map_indices_batch_id_vector, target_feat_map_indices)]
        src_ious = predicted_ious[(target_feat_map_indices_batch_id_vector, target_feat_map_indices)]
        loss_iou, loss_iou_aware = self.iou_loss(box_cxcywh_to_xyxy(src_boxes),
                                                 box_cxcywh_to_xyxy(target_bounding_box_label_matrix), src_ious)

        losses = {
            'loss_cls': self.cls_loss(predicted_class, target_class_label_vector),
            'loss_bbox': self.bbox_loss(src_boxes, target_bounding_box_label_matrix) / num_boxes_pos,
            'loss_iou': loss_iou / num_boxes_pos,
            'loss_iou_aware': loss_iou_aware / num_boxes_pos}
        return losses

    def forward(self, predicted, label):
        loss_dict = self._criterion(predicted, label)
        losses = sum(loss_dict[k] * self.weight_dict[k] for k in loss_dict.keys() if k in self.weight_dict)
        return (losses, *self._do_statistic(loss_dict))


def build_transt_criterion(train_config: dict):
    loss_parameters = train_config['train']['loss']['parameters']
    weight_dict = {'loss_cls': loss_parameters['target_classification']['gaussian_focal_loss']['weight'],
                   'loss_iou': loss_parameters['bounding_box_regression']['IoU']['weight'],
                   'loss_iou_aware': loss_parameters['quality_assessment']['IoU_aware']['weight']}
    cls_smooth_l1 = 'smooth_l1' in loss_parameters['bounding_box_regression']
    if cls_smooth_l1:
        weight_dict['loss_bbox'] = loss_parameters['bounding_box_regression']['smooth_l1']['weight']
    else:
        weight_dict['loss_bbox'] = loss_parameters['bounding_box_regression']['l1']['weight']

    return TransTExp1Criterion(weight_dict)
