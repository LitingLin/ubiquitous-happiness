import torch
import torch.distributed

from torch import nn
from Miscellaneous.torch.distributed import get_world_size, is_dist_available_and_initialized
from Miscellaneous.torch.distributed.reduce_dict import reduce_dict
from data.operator.bbox.spatial.vectorized.torch.cxcywh_to_xyxy import box_cxcywh_to_xyxy


class TransTCriterion(nn.Module):
    def __init__(self, weight_dict, cls_loss, bbox_loss, iou_loss):
        super().__init__()
        self.weight_dict = weight_dict
        self.cls_loss = cls_loss
        self.bbox_loss = bbox_loss
        if 'combined' in iou_loss:
            self.iou_and_iou_aware_combined_loss = iou_loss['combined']
        else:
            self.iou_and_iou_aware_combined_loss = None
            self.iou_loss = iou_loss['iou']
            if 'iou_aware' in iou_loss:
                self.iou_aware_loss = iou_loss['iou_aware']

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
        predicted_class = predicted[0]
        predicted_bounding_box = predicted[1]
        num_boxes_pos, target_feat_map_indices_batch_id_vector, target_feat_map_indices, target_class_label_vector, target_bounding_box_label_matrix = label
        if is_dist_available_and_initialized():
            torch.distributed.all_reduce(num_boxes_pos)
        num_boxes_pos = torch.clamp(num_boxes_pos / get_world_size(), min=1).item()

        losses = {}
        losses['loss_cls'] = self.cls_loss(predicted_class, target_class_label_vector)

        src_boxes = predicted_bounding_box[(target_feat_map_indices_batch_id_vector, target_feat_map_indices)]

        losses['loss_bbox'] = self.bbox_loss(src_boxes, target_bounding_box_label_matrix) / num_boxes_pos
        if self.iou_and_iou_aware_combined_loss is not None:
            predicted_ious = predicted[2]
            src_ious = predicted_ious[(target_feat_map_indices_batch_id_vector, target_feat_map_indices)]
            loss_iou, loss_iou_aware = self.iou_and_iou_aware_combined_loss(box_cxcywh_to_xyxy(src_boxes),
                                                     box_cxcywh_to_xyxy(target_bounding_box_label_matrix), src_ious)
            losses['loss_iou'] = loss_iou / num_boxes_pos
            losses['loss_iou_aware'] = loss_iou_aware / num_boxes_pos
        else:
            xyxy_src_boxes = box_cxcywh_to_xyxy(src_boxes)
            xyxy_box_labels = box_cxcywh_to_xyxy(target_bounding_box_label_matrix)
            losses['loss_iou'] = self.iou_loss(xyxy_src_boxes, xyxy_box_labels) / num_boxes_pos
            if self.iou_aware_loss is not None:
                predicted_ious = predicted[2]
                src_ious = predicted_ious[(target_feat_map_indices_batch_id_vector, target_feat_map_indices)]
                losses['loss_iou_aware'] = self.iou_aware_loss(xyxy_src_boxes, xyxy_box_labels, src_ious) / num_boxes_pos

        return losses

    def forward(self, predicted, label):
        loss_dict = self._criterion(predicted, label)
        losses = sum(loss_dict[k] * self.weight_dict[k] for k in loss_dict.keys() if k in self.weight_dict)
        return (losses, *self._do_statistic(loss_dict))
