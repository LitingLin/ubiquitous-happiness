from .criterion import SiamRPNLoss


def build_loss(config: dict):
    return SiamRPNLoss(config['TRAIN']['CLS_WEIGHT'], config['TRAIN']['LOC_WEIGHT'])
