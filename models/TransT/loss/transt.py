import torch
import torch.distributed

from torch import nn
from miscellanies.torch.distributed import get_world_size, is_dist_available_and_initialized
from miscellanies.torch.distributed.reduce_mean import reduce_mean_
from data.operator.bbox.spatial.vectorized.torch.cxcywh_to_xyxy import box_cxcywh_to_xyxy


class TransTCriterion(nn.Module):
    def __init__(self, cls_loss, bbox_loss, iou_loss):
        super().__init__()
        self.cls_loss = cls_loss
        self.bbox_loss = bbox_loss
        if 'combined' in iou_loss:
            self.iou_and_iou_aware_combined_loss = iou_loss['combined']
        else:
            self.iou_and_iou_aware_combined_loss = None
            self.iou_loss = iou_loss['iou']
            if 'iou_aware' in iou_loss:
                self.iou_aware_loss = iou_loss['iou_aware']
            else:
                self.iou_aware_loss = None

    def forward(self, predicted, label):
        predicted_class = predicted[0]
        predicted_bounding_box = predicted[1]
        num_boxes_pos, target_feat_map_indices_batch_id_vector, target_feat_map_indices, target_class_label_vector, target_bounding_box_label_matrix = label
        reduce_mean_(num_boxes_pos)
        num_boxes_pos = num_boxes_pos.item()
        num_boxes_pos = max(num_boxes_pos, 1)

        losses = []
        losses.append(self.cls_loss(predicted_class, target_class_label_vector))

        if target_feat_map_indices_batch_id_vector is not None:
            src_boxes = predicted_bounding_box[(target_feat_map_indices_batch_id_vector, target_feat_map_indices)]

            if self.bbox_loss is not None:
                losses.append(self.bbox_loss(src_boxes, target_bounding_box_label_matrix) / num_boxes_pos)
            if self.iou_and_iou_aware_combined_loss is not None:
                predicted_ious = predicted[2]
                src_ious = predicted_ious[(target_feat_map_indices_batch_id_vector, target_feat_map_indices)]
                loss_iou, loss_iou_aware = self.iou_and_iou_aware_combined_loss(box_cxcywh_to_xyxy(src_boxes),
                                                                                box_cxcywh_to_xyxy(
                                                                                    target_bounding_box_label_matrix),
                                                                                src_ious)
                losses.append(loss_iou / num_boxes_pos)
                losses.append(loss_iou_aware / num_boxes_pos)
            else:
                xyxy_src_boxes = box_cxcywh_to_xyxy(src_boxes)
                xyxy_box_labels = box_cxcywh_to_xyxy(target_bounding_box_label_matrix)
                if self.iou_loss is not None:
                    losses.append(self.iou_loss(xyxy_src_boxes, xyxy_box_labels) / num_boxes_pos)
                if self.iou_aware_loss is not None:
                    predicted_ious = predicted[2]
                    src_ious = predicted_ious[(target_feat_map_indices_batch_id_vector, target_feat_map_indices)]
                    losses.append(self.iou_aware_loss(xyxy_src_boxes, xyxy_box_labels, src_ious) / num_boxes_pos)
        else:
            zero = torch.mean(predicted_bounding_box * 0)
            if self.bbox_loss is not None:
                losses.append(zero)

            if self.iou_and_iou_aware_combined_loss is not None:
                losses.append(zero)
                losses.append(zero)
            else:
                if self.iou_loss is not None:
                    losses.append(zero)
                if self.iou_aware_loss is not None:
                    losses.append(zero)

        return losses


def _build_iou_loss(loss_parameters):
    combine_iou_loss_and_iou_aware_loss = False
    if 'combine_iou_loss_and_iou_aware_loss' in loss_parameters:
        combine_iou_loss_and_iou_aware_loss = loss_parameters['combine_iou_loss_and_iou_aware_loss']

    if not combine_iou_loss_and_iou_aware_loss:
        iou_loss_parameter = loss_parameters['IoU_loss']
        iou_loss_type = iou_loss_parameter['type']
        loss = {}
        if iou_loss_type == 'IoU':
            from models.loss.iou_loss import IoULoss
            iou_loss = IoULoss(reduction='sum')
        elif iou_loss_type == 'GIoU':
            from models.loss.iou_loss import GIoULoss
            iou_loss = GIoULoss(reduction='sum')
        elif iou_loss_type == 'DIoU':
            from models.loss.iou_loss import DIoULoss
            iou_loss = DIoULoss(reduction='sum')
        elif iou_loss_type == 'CIoU':
            from models.loss.iou_loss import CIoULoss
            iou_loss = CIoULoss(reduction='sum')
        else:
            raise RuntimeError(f'Unknown IoU type {iou_loss_type}')

        loss['iou'] = iou_loss

        if 'IoU_aware_loss' in loss_parameters:
            iou_aware_loss_parameter = loss_parameters['IoU_aware_loss']
            iou_aware_loss_type = iou_aware_loss_parameter['type']
            if iou_aware_loss_type == 'IoU':
                from models.loss.iou_aware_loss import IoUAwareLoss
                iou_aware_loss = IoUAwareLoss(reduction='sum')
            elif iou_aware_loss_type == 'GIoU':
                from models.loss.iou_aware_loss import GIoUAwareLoss
                iou_aware_loss = GIoUAwareLoss(reduction='sum')
            elif iou_aware_loss_type == 'DIoU':
                from models.loss.iou_aware_loss import DIoUAwareLoss
                iou_aware_loss = DIoUAwareLoss(reduction='sum')
            elif iou_aware_loss_type == 'CIoU':
                from models.loss.iou_aware_loss import CIoUAwareLoss
                iou_aware_loss = CIoUAwareLoss(reduction='sum')
            else:
                raise RuntimeError(f'Unknown IoU type {iou_aware_loss_type}')
            loss['iou_aware'] = iou_aware_loss
        return loss
    else:
        assert loss_parameters['IoU_loss']['type'] == loss_parameters['IoU_aware_loss']['type']
        iou_type = loss_parameters['IoU_loss']['type']
        if iou_type == 'IoU':
            from models.loss.iou_and_iou_aware_loss import IoUAndIoUAwareLoss
            return {'combined': IoUAndIoUAwareLoss(reduction='sum')}
        elif iou_type == 'GIoU':
            from models.loss.iou_and_iou_aware_loss import GIoUAndIoUAwareLoss
            return {'combined': GIoUAndIoUAwareLoss(reduction='sum')}
        elif iou_type == 'DIoU':
            from models.loss.iou_and_iou_aware_loss import GIoUAndIoUAwareLoss
            return {'combined': GIoUAndIoUAwareLoss(reduction='sum')}
        elif iou_type == 'CIoU':
            from models.loss.iou_and_iou_aware_loss import CIoUAndIoUAwareLoss
            return {'combined': CIoUAndIoUAwareLoss(reduction='sum')}
        else:
            raise RuntimeError(f'Unknown IoU type {iou_type}')



def build_transt_loss(train_config: dict):
    loss_parameters = train_config['optimization']['loss']

    if 'cross_entropy_loss' in loss_parameters:
        cls_loss_parameters = loss_parameters['cross_entropy_loss']
        weight = torch.ones(cls_loss_parameters['num_classes'] + 1)
        weight[-1] = cls_loss_parameters['background_class_weight']
        cls_loss = nn.CrossEntropyLoss(weight, reduction='mean')
    else:
        cls_loss = None

    if 'L1_loss' in loss_parameters:
        bbox_loss = nn.L1Loss(reduction='sum')
    elif 'smooth_L1_loss' in loss_parameters:
        smooth_L1_loss_parameter = loss_parameters['smooth_L1_loss']
        beta = 1.0
        if 'beta' in smooth_L1_loss_parameter:
            beta = smooth_L1_loss_parameter['beta']
        bbox_loss = nn.SmoothL1Loss(reduction='sum', beta=beta)
    else:
        bbox_loss = None

    iou_loss = _build_iou_loss(loss_parameters)
    return TransTCriterion(cls_loss, bbox_loss, iou_loss)
