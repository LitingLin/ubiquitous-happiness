import torch
import torch.nn as nn
import torch.distributed
from miscellanies.torch.distributed.reduce_mean import reduce_mean_
from models.TransT.loss._do_statistic import do_statistic
from data.operator.bbox.spatial.vectorized.torch.cxcywh_to_xyxy import box_cxcywh_to_xyxy


class GFocalMultiScaleCriterion(nn.Module):
    def __init__(self, quality_focal_loss, distribution_focal_loss, iou_loss,
                 quality_fn, weight_dict: dict, gfocal_reg_max: int,
                 strategy_min_quality_score: int, strategy_min_quality_score_drop_epoch: int):
        super(GFocalMultiScaleCriterion, self).__init__()
        self.quality_focal_loss = quality_focal_loss
        self.distribution_focal_loss = distribution_focal_loss
        self.iou_loss = iou_loss

        self.quality_fn = quality_fn

        self.weight_dict = weight_dict

        self.gfocal_reg_max = gfocal_reg_max

        self.epoch = 0
        self.strategy_min_quality_score = strategy_min_quality_score
        self.strategy_min_quality_score_drop_epoch = strategy_min_quality_score_drop_epoch

    def set_epoch(self, epoch):
        self.epoch = epoch

    def forward_single(self, num_positive_anchors, cls_score, predicted_bbox, bbox_distribution,
                       target_feat_map_indices_batch_id_vector, target_feat_map_indices,
                       target_class_label_vector, target_bounding_box_label_matrix):
        is_non_positives = target_feat_map_indices_batch_id_vector is None

        N, num_classes, H, W = cls_score.shape

        cls_score = cls_score.flatten(2).transpose(1, 2).view(N * H * W, -1)

        pos_inds = ((target_class_label_vector.view(-1) >= 0)
                    & (target_class_label_vector.view(-1) < num_classes)).nonzero().squeeze(1)

        if is_non_positives:
            weight_targets = torch.zeros([1], dtype=cls_score.dtype, device=cls_score.device)
        else:
            weight_targets = cls_score.detach()
            weight_targets = weight_targets.max(dim=1)[0][pos_inds]

        quality_score = torch.zeros((N, H * W), dtype=cls_score.dtype, device=cls_score.device)

        if not is_non_positives:
            predicted_bbox = predicted_bbox.view(N, H * W, 4)
            predicted_bbox = predicted_bbox[target_feat_map_indices_batch_id_vector, target_feat_map_indices]

            quality_score[target_feat_map_indices_batch_id_vector, target_feat_map_indices] = \
                self.quality_fn(box_cxcywh_to_xyxy(predicted_bbox.detach()),
                                box_cxcywh_to_xyxy(target_bounding_box_label_matrix))

        if self.strategy_min_quality_score_drop_epoch is not None and self.strategy_min_quality_score_drop_epoch >= self.epoch and self.strategy_min_quality_score > 0:
            quality_score.clamp_(min=self.strategy_min_quality_score, max=None)

        loss_cls = self.quality_focal_loss(cls_score, (target_class_label_vector.flatten(0), quality_score.flatten(0))) / num_positive_anchors

        if is_non_positives:
            loss_dfl = torch.mean(bbox_distribution * 0)
        else:
            bbox_distribution = bbox_distribution.view(N, H * W, 4 * (self.gfocal_reg_max + 1))
            bbox_distribution = bbox_distribution[target_feat_map_indices_batch_id_vector, target_feat_map_indices]
            loss_dfl = (self.distribution_focal_loss(bbox_distribution.view(-1, self.gfocal_reg_max + 1),
                                                     target_bounding_box_label_matrix.view(-1) * self.gfocal_reg_max) *
                        weight_targets[:, None].expand(-1, 4).reshape(-1)).sum() / 4

        if is_non_positives:
            loss_iou = torch.mean(predicted_bbox * 0)
        else:
            loss_iou = (self.iou_loss(box_cxcywh_to_xyxy(predicted_bbox),
                                      box_cxcywh_to_xyxy(target_bounding_box_label_matrix)) * weight_targets).sum()

        return loss_cls, loss_iou, loss_dfl, weight_targets.sum()

    def forward(self, predicted, label):
        # BBox format: normalized XYXY
        losses = {}
        '''
            cls_score: (N, C, H, W)
            predicted_bbox: (N, H, W, 4)
            bbox_distribution: (N, H, W, 4 * (reg_max + 1))
        '''
        cls_score_list, predicted_bbox_list, bbox_distribution_list = predicted

        num_positive_anchors, target_feat_map_indices_batch_id_vector_list, target_feat_map_indices_list, target_class_label_vector_list, target_bounding_box_label_matrix_list = label
        reduce_mean_(num_positive_anchors)
        num_positive_anchors = max(num_positive_anchors.item(), 1)

        loss_cls_list = []
        loss_iou_list = []
        loss_dfl_list = []
        weight_targets_sum_list = []

        for cls_score, predicted_bbox, bbox_distribution, \
            target_feat_map_indices_batch_id_vector, target_feat_map_indices, target_class_label_vector,\
            target_bounding_box_label_matrix in zip(cls_score_list, predicted_bbox_list, bbox_distribution_list, target_feat_map_indices_batch_id_vector_list, target_feat_map_indices_list, target_class_label_vector_list, target_bounding_box_label_matrix_list):
            loss_cls, loss_iou, loss_dfl, weight_targets_sum = self.forward_single(num_positive_anchors, cls_score, predicted_bbox, bbox_distribution,
            target_feat_map_indices_batch_id_vector, target_feat_map_indices, target_class_label_vector,
            target_bounding_box_label_matrix)
            loss_cls_list.append(loss_cls)
            loss_iou_list.append(loss_iou)
            loss_dfl_list.append(loss_dfl)
            weight_targets_sum_list.append(weight_targets_sum)

        loss_cls = sum(loss_cls_list)
        loss_iou = sum(loss_iou_list)
        loss_dfl = sum(loss_dfl_list)

        weight_targets_sum = sum(weight_targets_sum_list)

        reduce_mean_(weight_targets_sum)

        losses['loss_quality_focal'] = loss_cls / weight_targets_sum
        losses['loss_iou'] = loss_iou / weight_targets_sum
        losses['loss_distribution_focal'] = loss_dfl / weight_targets_sum

        loss = sum(losses[k] * self.weight_dict[k] for k in losses.keys())
        return (loss, *do_statistic(losses, self.weight_dict))


def build_multi_scale_gfocal_loss(network_config, train_config):
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

    qfl = QualityFocalLoss(False, loss_parameters['quality_focal_loss']['beta'], 'sum')
    dfl = DistributionFocalLoss('none')
    iou_loss_type = loss_parameters['iou_loss']['type']
    if iou_loss_type == 'IoU':
        from models.loss.iou_loss import IoULoss
        iou_loss = IoULoss(reduction='none')
    elif iou_loss_type == 'GIoU':
        from models.loss.iou_loss import GIoULoss
        iou_loss = GIoULoss(reduction='none')
    elif iou_loss_type == 'DIoU':
        from models.loss.iou_loss import DIoULoss
        iou_loss = DIoULoss(reduction='none')
    elif iou_loss_type == 'CIoU':
        from models.loss.iou_loss import CIoULoss
        iou_loss = CIoULoss(reduction='none')
    else:
        raise NotImplementedError(f'Unknown value: {iou_loss_type}')

    loss_weight = {
        'loss_quality_focal': loss_parameters['quality_focal_loss']['weight'],
        'loss_distribution_focal': loss_parameters['distribution_focal_loss']['weight'],
        'loss_iou': loss_parameters['iou_loss']['weight']
    }

    strategy_min_quality_score = 0
    strategy_min_quality_score_drop_epoch = None

    if 'advanced_strategy' in loss_parameters['quality_focal_loss']:
        qfl_advanced_strategy_parameters = loss_parameters['quality_focal_loss']['advanced_strategy']
        if 'min_quality_score' in qfl_advanced_strategy_parameters:
            strategy_min_quality_score = qfl_advanced_strategy_parameters['min_quality_score']['value']
            if 'drop_epoch' in qfl_advanced_strategy_parameters['min_quality_score']:
                strategy_min_quality_score_drop_epoch = qfl_advanced_strategy_parameters['min_quality_score']['drop_epoch']

    return GFocalMultiScaleCriterion(qfl, dfl, iou_loss, qa_func, loss_weight, gfocal_reg_max,
                                     strategy_min_quality_score, strategy_min_quality_score_drop_epoch)
