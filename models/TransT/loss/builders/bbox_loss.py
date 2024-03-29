from torch import nn


def build_bbox_loss(loss_parameters):
    if 'bounding_box_regression' not in loss_parameters:
        return None, {}
    if 'L1_loss' in loss_parameters['bounding_box_regression']:
        return nn.L1Loss(reduction='sum'), {'loss_bbox': loss_parameters['bounding_box_regression']['L1_loss']['weight']}
    elif 'smooth_L1_loss' in loss_parameters['bounding_box_regression']:
        smooth_L1_loss_parameter = loss_parameters['bounding_box_regression']['smooth_L1_loss']
        beta = 1.0
        if 'beta' in smooth_L1_loss_parameter:
            beta = smooth_L1_loss_parameter['beta']
        return nn.SmoothL1Loss(reduction='sum', beta=beta), {'loss_bbox': loss_parameters['bounding_box_regression']['smooth_L1_loss']['weight']}
    else:
        return None, {}
