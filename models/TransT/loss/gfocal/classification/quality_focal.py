def build_quality_focal(train_config: dict):
    from models.TransT.loss.single_class_quality_gfocal.quality_gfocal import QualityFocalLoss
    return QualityFocalLoss(False, train_config[''])