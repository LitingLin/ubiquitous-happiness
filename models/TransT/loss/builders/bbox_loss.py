from torch import nn


def build_bbox_loss(loss_parameters):
    if 'L1_loss' in loss_parameters['bounding_box_regression']:
        return nn.L1Loss(reduction='sum'), loss_parameters['bounding_box_regression']['L1_loss']['weight']
    elif 'smooth_L1_loss' in loss_parameters['bounding_box_regression']:
        return nn.SmoothL1Loss(reduction='sum'), loss_parameters['bounding_box_regression']['smooth_L1_loss']['weight']
    else:
        raise RuntimeError('Unknown bounding box loss')
