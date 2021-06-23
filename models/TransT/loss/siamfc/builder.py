import models.loss.siamfc.cross_entropy as cross_entropy_loss
from .criterion import SiamFCCriterion


def build_siamfc_loss(training_config: dict):
    loss_type = training_config['optimization']['loss']['type']
    if loss_type == 'Balanced':
        return SiamFCCriterion(cross_entropy_loss.BalancedLoss())
    elif loss_type == 'Focal':
        return SiamFCCriterion(cross_entropy_loss.FocalLoss())
    elif loss_type == 'GHMC':
        return SiamFCCriterion(cross_entropy_loss.GHMCLoss())
    elif loss_type == 'OHNM':
        return SiamFCCriterion(cross_entropy_loss.OHNMLoss())
    else:
        raise NotImplementedError
