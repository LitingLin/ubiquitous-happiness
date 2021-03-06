import models.neck.siamrpn.adjust


def build_neck(config: dict):
    if not config['ADJUST']['ADJUST']:
        return None
    return getattr(models.neck.siamrpn.adjust, config['ADJUST']['TYPE'])(**config['ADJUST']['KWARGS'])
