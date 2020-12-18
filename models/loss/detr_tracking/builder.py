from .criterion import DETRTrackingLoss


def build_detr_tracking_loss(train_config: dict):
    return DETRTrackingLoss(train_config['train']['loss']['weight'])
