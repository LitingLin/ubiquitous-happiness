import torch
from torch import nn


def build_cls_loss(loss_parameters):
    if 'cross_entropy_loss' in loss_parameters['classification']:
        cls_loss_parameters = loss_parameters['classification']['cross_entropy_loss']
        weight = torch.ones(cls_loss_parameters['num_classes'] + 1)
        weight[-1] = cls_loss_parameters['background_class_weight']
        return nn.CrossEntropyLoss(weight, reduction='mean'), {'loss_cls': cls_loss_parameters['weight']}
    elif 'gaussian_focal_loss' in loss_parameters['classification']:
        cls_loss_parameters = loss_parameters['classification']['gaussian_focal_loss']
        from models.loss.gaussian_focal_loss import GaussianFocalLoss
        return GaussianFocalLoss(), {'loss_cls': cls_loss_parameters['weight']}
    else:
        raise RuntimeError('Unknown classification loss parameters')
