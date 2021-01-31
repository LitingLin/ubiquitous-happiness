import models.loss.siamfc.cross_entropy as cross_entropy_loss


def build_siamfc_loss(training_config: dict):
    loss_type = training_config['model']['loss']['type']
    if loss_type == 'balanced':
        return cross_entropy_loss.BalancedLoss()
    elif loss_type == 'focal':
        return cross_entropy_loss.FocalLoss()
    elif loss_type == 'GHMC':
        return cross_entropy_loss.GHMCLoss()
    elif loss_type == 'OHNM':
        return cross_entropy_loss.OHNMLoss()
    else:
        raise NotImplementedError
