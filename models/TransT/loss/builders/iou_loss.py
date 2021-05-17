def build_iou_loss(loss_parameters):
    combine_iou_loss_and_iou_aware_loss = False
    if 'combine_iou_loss_and_iou_aware_loss' in loss_parameters:
        combine_iou_loss_and_iou_aware_loss = loss_parameters['combine_iou_loss_and_iou_aware_loss']

    if not combine_iou_loss_and_iou_aware_loss:
        iou_loss_parameter = loss_parameters['bounding_box_regression']['IoU_loss']
        iou_loss_type = iou_loss_parameter['type']
        loss = {}
        weight = {}
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
        weight['loss_iou'] = iou_loss_parameter['weight']

        if 'quality_assessment' in loss_parameters and 'IoU_aware_loss' in loss_parameters['quality_assessment']:
            iou_aware_loss_parameter = loss_parameters['quality_assessment']['IoU_aware_loss']
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
            weight['loss_iou_aware'] = iou_aware_loss_parameter['weight']
        return loss, weight
    else:
        assert loss_parameters['bounding_box_regression']['IoU_loss']['type'] == loss_parameters['quality_assessment']['IoU_aware_loss']['type']
        iou_type = loss_parameters['bounding_box_regression']['IoU_loss']['type']
        loss_weight = {'loss_iou': loss_parameters['bounding_box_regression']['IoU_loss']['weight'], 'loss_iou_aware': loss_parameters['quality_assessment']['IoU_aware_loss']['weight']}
        if iou_type == 'IoU':
            from models.loss.iou_and_iou_aware_loss import IoUAndIoUAwareLoss
            return {'combined': IoUAndIoUAwareLoss(reduction='sum')}, loss_weight
        elif iou_type == 'GIoU':
            from models.loss.iou_and_iou_aware_loss import GIoUAndIoUAwareLoss
            return {'combined': GIoUAndIoUAwareLoss(reduction='sum')}, loss_weight
        elif iou_type == 'DIoU':
            from models.loss.iou_and_iou_aware_loss import GIoUAndIoUAwareLoss
            return {'combined': GIoUAndIoUAwareLoss(reduction='sum')}, loss_weight
        elif iou_type == 'CIoU':
            from models.loss.iou_and_iou_aware_loss import CIoUAndIoUAwareLoss
            return {'combined': CIoUAndIoUAwareLoss(reduction='sum')}, loss_weight
        else:
            raise RuntimeError(f'Unknown IoU type {iou_type}')
