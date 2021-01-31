from models.network.siamfc.builder import build_siamfc_network
from models.network.siamfc.initializer import get_siamfc_model_initialization_function
from models.loss.siamfc.builder import build_siamfc_loss
import torch
from torch import optim
import numpy as np


def _get_optimizer_learnable_params(model, optimizer_config):
    lr = optimizer_config['initial_lr']
    weight_decay = optimizer_config['weight_decay']
    if 'layer_wise' in optimizer_config:
        backbone = model.backbone
        head = model.head

        layer_wise_config = optimizer_config['layer_wise']

        branch_conv_weight_lr_ratio = layer_wise_config['lr']['conv']['weight']
        branch_conv_bias_lr_ratio = layer_wise_config['lr']['conv']['bias']
        branch_bn_gamma_lr_ratio = layer_wise_config['lr']['bn']['gamma']
        branch_bn_beta_lr_ratio = layer_wise_config['lr']['bn']['beta']

        branch_conv_weight_weight_decay = layer_wise_config['weight_decay']['conv']['weight']
        branch_conv_bias_weight_decay = layer_wise_config['weight_decay']['conv']['bias']
        branch_bn_gamma_weight_decay = layer_wise_config['weight_decay']['bn']['gamma']
        branch_bn_beta_weight_decay = layer_wise_config['weight_decay']['bn']['beta']

        head_adjust_gain_lr_ratio = layer_wise_config['lr']['head']['gain']
        head_adjust_bias_lr_ratio = layer_wise_config['lr']['head']['bias']

        head_adjust_gain_weight_decay = layer_wise_config['weight_decay']['head']['gain']
        head_adjust_bias_weight_decay = layer_wise_config['weight_decay']['head']['bias']

        optimizer_params = []


        optimizer_params = [
            {'params': backbone.conv1[0].weight, 'lr': lr * branch_conv_weight_lr_ratio,
             'weight_decay': weight_decay * branch_conv_weight_weight_decay},
            {'params': backbone.conv1[0].bias, 'lr': lr * branch_conv_bias_lr_ratio,
             'weight_decay': weight_decay * branch_conv_bias_weight_decay},
            {'params': backbone.conv1[1].weight, 'lr': lr * branch_bn_gamma_lr_ratio,
             'weight_decay': weight_decay * branch_bn_gamma_weight_decay},
            {'params': backbone.conv1[1].bias, 'lr': lr * branch_bn_beta_lr_ratio,
             'weight_decay': weight_decay * branch_bn_beta_weight_decay},
            {'params': backbone.conv2[0].weight, 'lr': lr * branch_conv_weight_lr_ratio,
             'weight_decay': base_weight_decay * branch_conv_weight_weight_decay},
            {'params': backbone.conv2[0].bias, 'lr': lr * branch_conv_bias_lr_ratio,
             'weight_decay': base_weight_decay * branch_conv_bias_weight_decay},
            {'params': backbone.conv2[1].weight, 'lr': lr * branch_bn_gamma_lr_ratio,
             'weight_decay': base_weight_decay * branch_bn_gamma_weight_decay},
            {'params': backbone.conv2[1].bias, 'lr': lr * branch_bn_beta_lr_ratio,
             'weight_decay': base_weight_decay * branch_bn_beta_weight_decay},
            {'params': backbone.conv3[0].weight, 'lr': lr * branch_conv_weight_lr_ratio,
             'weight_decay': base_weight_decay * branch_conv_weight_weight_decay},
            {'params': backbone.conv3[0].bias, 'lr': lr * branch_conv_bias_lr_ratio,
             'weight_decay': base_weight_decay * branch_conv_bias_weight_decay},
            {'params': backbone.conv3[1].weight, 'lr': lr * branch_bn_gamma_lr_ratio,
             'weight_decay': base_weight_decay * branch_bn_gamma_weight_decay},
            {'params': backbone.conv3[1].bias, 'lr': lr * branch_bn_beta_lr_ratio,
             'weight_decay': base_weight_decay * branch_bn_beta_weight_decay},
            {'params': backbone.conv4[0].weight, 'lr': lr * branch_conv_weight_lr_ratio,
             'weight_decay': base_weight_decay * branch_conv_weight_weight_decay},
            {'params': backbone.conv4[0].bias, 'lr': lr * branch_conv_bias_lr_ratio,
             'weight_decay': base_weight_decay * branch_conv_bias_weight_decay},
            {'params': backbone.conv4[1].weight, 'lr': lr * branch_bn_gamma_lr_ratio,
             'weight_decay': base_weight_decay * branch_bn_gamma_weight_decay},
            {'params': backbone.conv4[1].bias, 'lr': lr * branch_bn_beta_lr_ratio,
             'weight_decay': base_weight_decay * branch_bn_beta_weight_decay},
            {'params': backbone.conv5[0].weight, 'lr': lr * branch_conv_weight_lr_ratio,
             'weight_decay': base_weight_decay * branch_conv_weight_weight_decay},
            {'params': backbone.conv5[0].bias, 'lr': lr * branch_conv_bias_lr_ratio,
             'weight_decay': base_weight_decay * branch_conv_bias_weight_decay},

            {'params': head.adjust_gain, 'lr': base_lr * head_adjust_gain_lr_ratio,
             'weight_decay': base_weight_decay * head_adjust_gain_weight_decay},
            {'params': head.adjust_bias, 'lr': base_lr * head_adjust_bias_lr_ratio,
             'weight_decay': base_weight_decay * head_adjust_bias_weight_decay},
        ]
    else:
        optimizer_params = model.parameters()


def _setup_siamfc_optimizer(model, train_config):
    optimizer_config = train_config['train']['optimizer']
    if optimizer_config['type'] == 'sgd':
        optimizer = optim.SGD()

    gamma = np.power(
            optimizer_config['ultimate_lr'] / optimizer_config['initial_lr'],
            1.0 / train_config['train']['epochs'])
    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma)



def build_siamfc_training_actor(args, network_config: dict, train_config: dict, epoch_changed_event_signal_slots=None):
    model = build_siamfc_network(network_config)
    device = torch.device(args.device)

    criterion = build_siamfc_loss(train_config)

    optimizer, lr_scheduler = _setup_siamfc_optimizer(model, train_config)

    model.to(device)
    criterion.to(device)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    return DETRTrackingActor(model, criterion, optimizer, lr_scheduler, initialize_siam_encoder_detr_track, epoch_changed_event_signal_slots)
