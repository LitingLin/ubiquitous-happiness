def iou_loss_data_adaptor(pred, label):


def build_IoU(loss_parameters: dict):
    iou_loss_type = loss_parameters['type']
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
    return iou_loss