from models.network.siamfc.builder import build_siamfc_network
from models.network.siamfc.initializer import get_siamfc_model_initialization_function
from models.loss.siamfc.builder import build_siamfc_loss
import torch
from torch import optim
import numpy as np
from Miscellaneous.nullable_get import nullable_get
from models.backbone.siamfc.alexnet import *
from models.head.siamfc.siamfc import *
from .actor import SiamFCTrainingActor
from data.siamfc.label import SiamFCLabelGenerator
from data.siamfc.processor.siamfc import SiamFCDataProcessor
from data.siamfc.post_combiner import SiamFCPostDataCombiner
from data.siamfc.dataset import build_tracking_dataset
from data.torch.data_loader import build_torch_train_val_dataloader


def _get_optimizer_learnable_params(model, optimizer_config):
    lr = optimizer_config['initial_lr']
    weight_decay = optimizer_config['weight_decay']
    if 'layer_wise' in optimizer_config:
        backbone = model.backbone
        head = model.head
        assert isinstance(backbone, (AlexNetV1, AlexNetV2, AlexNetV3))
        assert isinstance(head, (SiamFCLinearHead, SiamFCBNHead))

        layer_wise_config = optimizer_config['layer_wise']

        branch_conv_weight_lr_ratio = layer_wise_config['lr']['conv']['weight']
        branch_conv_bias_lr_ratio = layer_wise_config['lr']['conv']['bias']
        branch_bn_gamma_lr_ratio = layer_wise_config['lr']['bn']['gamma']
        branch_bn_beta_lr_ratio = layer_wise_config['lr']['bn']['beta']

        branch_conv_weight_weight_decay_ratio = layer_wise_config['weight_decay']['conv']['weight']
        branch_conv_bias_weight_decay_ratio = layer_wise_config['weight_decay']['conv']['bias']
        branch_bn_gamma_weight_decay_ratio = layer_wise_config['weight_decay']['bn']['gamma']
        branch_bn_beta_weight_decay_ratio = layer_wise_config['weight_decay']['bn']['beta']

        head_adjust_gain_lr_ratio = nullable_get(layer_wise_config, ('lr', 'head', 'gain'))
        head_adjust_bias_lr_ratio = nullable_get(layer_wise_config, ('lr', 'head', 'bias'))
        head_adjust_gain_weight_decay_ratio = nullable_get(layer_wise_config, ('weight_decay', 'head', 'gain'))
        head_adjust_bias_weight_decay_ratio = nullable_get(layer_wise_config, ('weight_decay', 'head', 'bias'))

        head_bn_gamma_lr_ratio = layer_wise_config['lr']['head']['gain']
        head_bn_beta_lr_ratio = layer_wise_config['lr']['head']['bias']

        head_bn_gamma_weight_decay_ratio = nullable_get(layer_wise_config, ('weight_decay', 'head', 'gain'))
        head_bn_beta_weight_decay_ratio = nullable_get(layer_wise_config, ('weight_decay', 'head', 'bias'))

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

        if isinstance(head, SiamFCLinearHead):
            optimizer_params.append(
                _generate_params_dict(head.adjust_gain, head_adjust_gain_lr_ratio, head_adjust_gain_weight_decay_ratio))
            optimizer_params.append(
                _generate_params_dict(head.adjust_bias, head_adjust_bias_lr_ratio, head_adjust_bias_weight_decay_ratio))
        if isinstance(head, SiamFCBNHead):
            optimizer_params.append(
                _generate_params_dict(head.adjust_bn.weight, head_bn_gamma_lr_ratio, head_bn_gamma_weight_decay_ratio))
            optimizer_params.append(
                _generate_params_dict(head.adjust_bn.bias, head_bn_beta_lr_ratio, head_bn_beta_weight_decay_ratio))
        return optimizer_params
    else:
        return {'params': model.parameters(), 'lr': lr, 'weight_decay': weight_decay}


def _setup_siamfc_optimizer(model, train_config):
    optimizer_config = train_config['train']['optimizer']
    if optimizer_config['type'] == 'SGD':
        lr = optimizer_config['initial_lr']
        weight_decay = optimizer_config['weight_decay']
        momentum = optimizer_config['momentum']
        optimizer = optim.SGD(_get_optimizer_learnable_params(model, optimizer_config), lr=lr, weight_decay=weight_decay, momentum=momentum)
    else:
        raise NotImplementedError

    gamma = np.power(
            optimizer_config['ultimate_lr'] / optimizer_config['initial_lr'],
            1.0 / train_config['train']['epochs'])
    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma)
    return optimizer, lr_scheduler


def build_siamfc_training_actor(args, network_config: dict, train_config: dict, epoch_changed_event_signal_slots=None):
    model = build_siamfc_network(network_config)
    device = torch.device(args.device)

    criterion = build_siamfc_loss(train_config)

    optimizer, lr_scheduler = _setup_siamfc_optimizer(model, train_config)

    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model.to(device)
    criterion.to(device)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    return SiamFCTrainingActor(model, criterion, optimizer, lr_scheduler, get_siamfc_model_initialization_function(train_config), epoch_changed_event_signal_slots)


def _build_dataloader(args, train_config: dict, train_dataset_config_path: str, val_dataset_config_path: str):
    data_config = train_config['data']

    label_generator = SiamFCLabelGenerator(data_config['label']['size'], data_config['label']['r_pos'], data_config['label']['r_neg'], train_config['model']['total_stride'])

    siamfc_processor = SiamFCDataProcessor(data_config['exemplar_sz'], data_config['instance_sz'], data_config['context'],
                                           data_config['augmentation']['translation'], data_config['augmentation']['stretch_ratio'],
                                           data_config['augmentation']['rgb_variance_z_crop'],
                                           data_config['augmentation']['rgb_variance_x_crop'],
                                           data_config['augmentation']['random_gray_ratio'])

    siamfc_data_processor = SiamFCPostDataCombiner(siamfc_processor, label_generator)

    train_dataset, val_dataset = build_tracking_dataset(train_config, train_dataset_config_path, val_dataset_config_path, siamfc_data_processor, siamfc_data_processor)

    epoch_changed_event_signal_slots = []

    data_loader_train, data_loader_val = build_torch_train_val_dataloader(train_dataset, val_dataset,
                                                                          train_config['train']['batch_size'],
                                                                          train_config['val']['batch_size'],
                                                                          args.num_workers, args.num_workers,
                                                                          args.device, args.distributed,
                                                                          epoch_changed_event_signal_slots)

    return data_loader_train, data_loader_val, epoch_changed_event_signal_slots


def build_training_actor_and_dataloader(args, network_config: dict, train_config: dict, train_dataset_config_path: str, val_dataset_config_path: str):
    data_loader_train, data_loader_val, epoch_changed_event_signal_slots = _build_dataloader(args, train_config, train_dataset_config_path, val_dataset_config_path)

    actor = build_siamfc_training_actor(args, network_config, train_config, epoch_changed_event_signal_slots)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        actor.load_state_dict(checkpoint)
        args.start_epoch = actor.get_epoch()
    else:
        actor.reset_parameters()

    return actor, data_loader_train, data_loader_val
