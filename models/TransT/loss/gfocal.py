import torch
import torch.nn as nn
import torch.distributed
from miscellanies.torch.distributed import is_dist_available_and_initialized, get_world_size
from ._do_statistic import do_statistic


class GFocalCriterion(nn.Module):
    def __init__(self, quality_focal_loss, distribution_focal_loss, iou_loss, quality_fn, weight_dict: dict, gfocal_reg_max: int):
        super(GFocalCriterion, self).__init__()
        self.quality_focal_loss = quality_focal_loss
        self.distribution_focal_loss = distribution_focal_loss
        self.iou_loss = iou_loss

        self.quality_fn = quality_fn

        self.weight_dict = weight_dict

        self.gfocal_reg_max = gfocal_reg_max

    def forward(self, predicted, label):
        # BBox format: normalized XYXY

        losses = {}
        '''
            cls_score: (N, C, H, W)
            predicted_bbox: (N, H, W, 4)
            bbox_distribution: (N, H, W, 4 * (reg_max + 1))
        '''
        cls_score, predicted_bbox, bbox_distribution = predicted

        num_boxes_pos, target_feat_map_indices_batch_id_vector, target_feat_map_indices, target_class_label_vector, target_bounding_box_label_matrix = label
        if is_dist_available_and_initialized():
            torch.distributed.all_reduce(num_boxes_pos)

        num_boxes_pos = (num_boxes_pos / get_world_size()).item()

        is_non_positives = target_feat_map_indices_batch_id_vector is None

        N, _, H, W = cls_score.shape
        quality_score = torch.zeros((N, H * W), dtype=cls_score.dtype, device=cls_score.device)

        if not is_non_positives:
            predicted_bbox = predicted_bbox.view(N, H * W, 4)
            predicted_bbox = predicted_bbox[target_feat_map_indices_batch_id_vector, target_feat_map_indices]
            valid_mask = (predicted_bbox[:, 2] - predicted_bbox[:, 0] > 0) & (predicted_bbox[:, 3] - predicted_bbox[:, 1] > 0)
            predicted_bbox_ = predicted_bbox[valid_mask]
            if len(predicted_bbox_) != 0:
                quality_score[target_feat_map_indices_batch_id_vector[valid_mask], target_feat_map_indices[valid_mask]] = \
                    self.quality_fn(predicted_bbox_.detach(), target_bounding_box_label_matrix[valid_mask])

        losses['loss_quality_focal'] = self.quality_focal_loss(cls_score.flatten(2).transpose(1, 2).view(N * H * W, -1), (target_class_label_vector.flatten(0), quality_score.flatten(0)))

        if is_non_positives:
            losses['loss_distribution_focal'] = torch.mean(bbox_distribution * 0)
        else:
            bbox_distribution = bbox_distribution.view(N, H * W, 4 * (self.gfocal_reg_max + 1))
            bbox_distribution = bbox_distribution[target_feat_map_indices_batch_id_vector, target_feat_map_indices]
            losses['loss_distribution_focal'] = self.distribution_focal_loss(bbox_distribution.view(-1, self.gfocal_reg_max + 1), target_bounding_box_label_matrix.view(-1) * self.gfocal_reg_max) / num_boxes_pos

        if is_non_positives:
            losses['loss_iou'] = torch.mean(predicted_bbox * 0)
        else:
            if len(predicted_bbox_) == 0:
                losses['loss_iou'] = torch.mean(predicted_bbox * 0)
            else:
                losses['loss_iou'] = self.iou_loss(predicted_bbox_, target_bounding_box_label_matrix[valid_mask]) / num_boxes_pos

        loss = sum(losses[k] * self.weight_dict[k] for k in losses.keys())
        return (loss, *do_statistic(losses, self.weight_dict))


def build_gfocal_loss(network_config, train_config):
    loss_parameters = train_config['optimization']['loss']
    qfl_parameters = loss_parameters['quality_focal_loss']
    quality_function = qfl_parameters['quality_function']
    if quality_function == 'IoU':
        from models.operator.iou.iou import iou
        qa_func = iou
    elif quality_function == ['GIoU']:
        from models.operator.iou.giou import giou
        qa_func = giou
    elif quality_function == ['GIoU']:
        from models.operator.iou.giou import giou
        qa_func = giou
    elif quality_function == ['DIoU']:
        from models.operator.iou.diou import diou
        qa_func = diou
    elif quality_function == ['CIoU']:
        from models.operator.iou.ciou import ciou
        qa_func = ciou
    else:
        raise NotImplementedError(f'Unknown value: {quality_function}')

    gfocal_reg_max = network_config['head']['parameters']['reg_max']

    from models.loss.gfocal_v2 import QualityFocalLoss, DistributionFocalLoss

    qfl = QualityFocalLoss(False, loss_parameters['quality_focal_loss']['beta'], 'mean')
    dfl = DistributionFocalLoss('sum')
    iou_loss_type = loss_parameters['iou_loss']['type']
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
        raise NotImplementedError(f'Unknown value: {iou_loss_type}')

    loss_weight = {
        'loss_quality_focal': loss_parameters['quality_focal_loss']['weight'],
        'loss_distribution_focal': loss_parameters['distribution_focal_loss']['weight'],
        'loss_iou': loss_parameters['iou_loss']['weight']
    }
    return GFocalCriterion(qfl, dfl, iou_loss, qa_func, loss_weight, gfocal_reg_max)
