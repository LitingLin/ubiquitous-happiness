class DistributionFocalDataAdaptor:
    def __init__(self, gfocal_reg_max):
        self.gfocal_reg_max = gfocal_reg_max

    def __call__(self, pred, label, _):
        (cls_score, predicted_bbox, bbox_distribution) = pred
        if label is None:
            return False, bbox_distribution.sum() * 0

        num_boxes_pos, target_bounding_box_label_matrix = label
        return True, (bbox_distribution.view(-1, self.gfocal_reg_max + 1), target_bounding_box_label_matrix.view(-1) * self.gfocal_reg_max)


def reduce_by_weight(loss, pred, label, context):
    return (loss * context['sample_weight'][:, None].expand(-1, 4).reshape(-1)).sum() / 4


def build_distribution_focal(loss_parameters, network_parameters, _):
    from models.loss.gfocal_v2 import DistributionFocalLoss
    if 'reduce' in loss_parameters and loss_parameters['reduce'] == 'weighted':
        loss_reduce_function = reduce_by_weight
    else:
        from models.TransT.loss.common.reduction.default import build_loss_reduction_function
        loss_reduce_function = build_loss_reduction_function(loss_parameters)
    return DistributionFocalLoss(reduction='none'), \
           DistributionFocalDataAdaptor(network_parameters['head']['parameters']['reg_max']), loss_reduce_function
