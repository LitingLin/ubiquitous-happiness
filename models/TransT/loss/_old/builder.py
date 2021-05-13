from models.TransT.loss.transt import TransTCriterion


def build_transt_criterion_old(train_config: dict):
    loss_parameters = train_config['train']['loss']
    weight_dict = {'loss_ce': loss_parameters['weight']['cross_entropy'], 'loss_bbox': loss_parameters['weight']['bbox'], 'loss_giou': loss_parameters['weight']['giou']}
    return TransTCriterion(weight_dict, loss_parameters['eos_coef'])
