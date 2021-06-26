import torch.optim
from models.TransT.neck.classical import SiamFCLinearNeck, SiamFCBNNeck


def _get_optimizer_learnable_params(model, optimizer_config):
    lr = optimizer_config['lr']
    weight_decay = optimizer_config['weight_decay']
    if 'layer_wise' in optimizer_config:
        backbone = model.backbone
        neck = model.neck
        assert isinstance(neck, (SiamFCLinearNeck, SiamFCBNNeck))

        layer_wise_config = optimizer_config['layer_wise']

        branch_conv_weight_lr_ratio = layer_wise_config['lr']['backbone']['conv']['weight']
        branch_conv_bias_lr_ratio = layer_wise_config['lr']['backbone']['conv']['bias']
        branch_bn_gamma_lr_ratio = layer_wise_config['lr']['backbone']['bn']['gamma']
        branch_bn_beta_lr_ratio = layer_wise_config['lr']['backbone']['bn']['beta']

        branch_conv_weight_weight_decay_ratio = layer_wise_config['weight_decay']['backbone']['conv']['weight']
        branch_conv_bias_weight_decay_ratio = layer_wise_config['weight_decay']['backbone']['conv']['bias']
        branch_bn_gamma_weight_decay_ratio = layer_wise_config['weight_decay']['backbone']['bn']['gamma']
        branch_bn_beta_weight_decay_ratio = layer_wise_config['weight_decay']['backbone']['bn']['beta']

        optimizer_params = [
            {'params': backbone.conv1[0].weight, 'lr': lr * branch_conv_weight_lr_ratio,
             'weight_decay': weight_decay * branch_conv_weight_weight_decay_ratio},
            {'params': backbone.conv1[0].bias, 'lr': lr * branch_conv_bias_lr_ratio,
             'weight_decay': weight_decay * branch_conv_bias_weight_decay_ratio},
            {'params': backbone.conv1[1].weight, 'lr': lr * branch_bn_gamma_lr_ratio,
             'weight_decay': weight_decay * branch_bn_gamma_weight_decay_ratio},
            {'params': backbone.conv1[1].bias, 'lr': lr * branch_bn_beta_lr_ratio,
             'weight_decay': weight_decay * branch_bn_beta_weight_decay_ratio},
            {'params': backbone.conv2[0].weight, 'lr': lr * branch_conv_weight_lr_ratio,
             'weight_decay': weight_decay * branch_conv_weight_weight_decay_ratio},
            {'params': backbone.conv2[0].bias, 'lr': lr * branch_conv_bias_lr_ratio,
             'weight_decay': weight_decay * branch_conv_bias_weight_decay_ratio},
            {'params': backbone.conv2[1].weight, 'lr': lr * branch_bn_gamma_lr_ratio,
             'weight_decay': weight_decay * branch_bn_gamma_weight_decay_ratio},
            {'params': backbone.conv2[1].bias, 'lr': lr * branch_bn_beta_lr_ratio,
             'weight_decay': weight_decay * branch_bn_beta_weight_decay_ratio},
            {'params': backbone.conv3[0].weight, 'lr': lr * branch_conv_weight_lr_ratio,
             'weight_decay': weight_decay * branch_conv_weight_weight_decay_ratio},
            {'params': backbone.conv3[0].bias, 'lr': lr * branch_conv_bias_lr_ratio,
             'weight_decay': weight_decay * branch_conv_bias_weight_decay_ratio},
            {'params': backbone.conv3[1].weight, 'lr': lr * branch_bn_gamma_lr_ratio,
             'weight_decay': weight_decay * branch_bn_gamma_weight_decay_ratio},
            {'params': backbone.conv3[1].bias, 'lr': lr * branch_bn_beta_lr_ratio,
             'weight_decay': weight_decay * branch_bn_beta_weight_decay_ratio},
            {'params': backbone.conv4[0].weight, 'lr': lr * branch_conv_weight_lr_ratio,
             'weight_decay': weight_decay * branch_conv_weight_weight_decay_ratio},
            {'params': backbone.conv4[0].bias, 'lr': lr * branch_conv_bias_lr_ratio,
             'weight_decay': weight_decay * branch_conv_bias_weight_decay_ratio},
            {'params': backbone.conv4[1].weight, 'lr': lr * branch_bn_gamma_lr_ratio,
             'weight_decay': weight_decay * branch_bn_gamma_weight_decay_ratio},
            {'params': backbone.conv4[1].bias, 'lr': lr * branch_bn_beta_lr_ratio,
             'weight_decay': weight_decay * branch_bn_beta_weight_decay_ratio},
            {'params': backbone.conv5[0].weight, 'lr': lr * branch_conv_weight_lr_ratio,
             'weight_decay': weight_decay * branch_conv_weight_weight_decay_ratio},
            {'params': backbone.conv5[0].bias, 'lr': lr * branch_conv_bias_lr_ratio,
             'weight_decay': weight_decay * branch_conv_bias_weight_decay_ratio}
        ]

        def _generate_params_dict(params, lr_ratio, weight_decay_ratio):
            dict_ = {'params': params}
            if lr_ratio is not None:
                dict_['lr'] = lr * lr_ratio
            if weight_decay_ratio is not None:
                dict_['weight_decay'] = weight_decay * weight_decay_ratio
            return dict_

        if isinstance(neck, SiamFCLinearNeck):
            neck_adjust_gain_lr_ratio = layer_wise_config['lr']['neck']['gain']
            neck_adjust_bias_lr_ratio = layer_wise_config['lr']['neck']['bias']
            neck_adjust_gain_weight_decay_ratio = layer_wise_config['weight_decay']['neck']['gain']
            neck_adjust_bias_weight_decay_ratio = layer_wise_config['weight_decay']['neck']['bias']
            optimizer_params.append(
                _generate_params_dict(neck.adjust_gain, neck_adjust_gain_lr_ratio, neck_adjust_gain_weight_decay_ratio))
            optimizer_params.append(
                _generate_params_dict(neck.adjust_bias, neck_adjust_bias_lr_ratio, neck_adjust_bias_weight_decay_ratio))
        if isinstance(neck, SiamFCBNNeck):
            neck_bn_gamma_lr_ratio = layer_wise_config['lr']['neck']['gamma']
            neck_bn_beta_lr_ratio = layer_wise_config['lr']['neck']['beta']
            neck_bn_gamma_weight_decay_ratio = layer_wise_config['weight_decay']['neck']['gamma']
            neck_bn_beta_weight_decay_ratio = layer_wise_config['weight_decay']['neck']['beta']

            optimizer_params.append(
                _generate_params_dict(neck.adjust_bn.weight, neck_bn_gamma_lr_ratio, neck_bn_gamma_weight_decay_ratio))
            optimizer_params.append(
                _generate_params_dict(neck.adjust_bn.bias, neck_bn_beta_lr_ratio, neck_bn_beta_weight_decay_ratio))
        return optimizer_params
    else:
        return model.parameters()


def build_siamfc_layerwise_optimizer(model, train_config):
    optimizer_config = train_config['optimization']['optimizer']
    return torch.optim.SGD(_get_optimizer_learnable_params(model, optimizer_config), lr=optimizer_config['lr'], weight_decay=optimizer_config['weight_decay'], momentum=optimizer_config['momentum'])
