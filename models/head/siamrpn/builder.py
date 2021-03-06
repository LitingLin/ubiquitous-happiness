import models.head.siamrpn.rpn


def build_rpn(config: dict):
    rpn_type = config['RPN']['TYPE']
    rpn_params = config['RPN']['KWARGS']
    return getattr(models.head.siamrpn.rpn, rpn_type)(**rpn_params)
