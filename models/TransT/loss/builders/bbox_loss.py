from torch import nn


def build_bbox_loss(loss_parameters):
    if 'L1_loss' in loss_parameters['bounding_box_regression']:
        return nn.L1Loss(reduction='sum'), {'loss_iou': loss_parameters['bounding_box_regression']['L1_loss']['weight']}
    elif 'smooth_L1_loss' in loss_parameters['bounding_box_regression']:
        smooth_L1_loss_parameter = loss_parameters['bounding_box_regression']['smooth_L1_loss']
        beta = 1.0
        if 'beta' in smooth_L1_loss_parameter:
            beta = smooth_L1_loss_parameter['beta']
        return nn.SmoothL1Loss(reduction='sum', beta=beta), {'loss_iou': loss_parameters['bounding_box_regression']['smooth_L1_loss']['weight']}
    else:
        raise RuntimeError('Unknown bounding box loss')
