import torch
from models.network.detr_tracking_variants.siam_encoder.builder import build_siam_encoder_detr_track, \
    initialize_siam_encoder_detr_track
from models.loss.detr_tracking.builder import build_detr_tracking_loss
from training.detr_tracking.actor import DETRTrackingActor
from data.detr_tracking_variants.siam_encoder.processor.mask_generator import SiamTransformerMaskGeneratingProcessor
from data.siamfc.processor.z_curate_x_resize import SiamFC_Z_Curate_BBOX_XYWH_X_Resize_BBOX_CXCYWHNormalized_Processor
from data.torch.data_loader import build_torch_train_val_dataloader
from data.siamfc.dataset import build_tracking_dataset


def _setup_optimizer(model, train_config):
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": train_config['train']['lr_backbone']
        },
    ]

    if train_config['train']['optimizer'] == 'AdamW':
        optimizer = torch.optim.AdamW(param_dicts, lr=train_config['train']['lr'],
                                      weight_decay=train_config['train']['weight_decay'])
    elif train_config['train']['optimizer'] == 'SGD':
        optimizer = torch.optim.SGD(param_dicts, lr=train_config['train']['lr'], momentum=0.9,
                                    weight_decay=train_config['train']['weight_decay'])
    else:
        raise Exception(f"unknown optimizer {train_config['train']['optimizer']}")
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, train_config['train']['lr_drop'])
    return optimizer, lr_scheduler


def build_siam_encoder_detr_tracking_training_actor(args, net_config: dict, train_config: dict,
                                                    epoch_changed_event_signal_slots=None):
    model = build_siam_encoder_detr_track(net_config)
    device = torch.device(args.device)

    criterion = build_detr_tracking_loss(train_config)
    optimizer, lr_scheduler = _setup_optimizer(model, train_config)

    model.to(device)
    criterion.to(device)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    return DETRTrackingActor(model, criterion, optimizer, lr_scheduler, initialize_siam_encoder_detr_track,
                             epoch_changed_event_signal_slots)


def _build_dataloader(args, network_config: dict, train_config: dict, train_dataset_config_path: str,
                      val_dataset_config_path: str):
    siamfc_like_processor = SiamFC_Z_Curate_BBOX_XYWH_X_Resize_BBOX_CXCYWHNormalized_Processor(
        network_config['backbone']['siamfc']['exemplar_size'], network_config['backbone']['siamfc']['instance_size'],
        network_config['backbone']['siamfc']['context'])
    processor = SiamTransformerMaskGeneratingProcessor(siamfc_like_processor)

    train_dataset, val_dataset = build_tracking_dataset(train_config, train_dataset_config_path,
                                                        val_dataset_config_path, processor, processor)

    epoch_changed_event_signal_slots = []

    data_loader_train, data_loader_val = build_torch_train_val_dataloader(train_dataset, val_dataset,
                                                                          train_config['train']['batch_size'],
                                                                          train_config['val']['batch_size'],
                                                                          args.num_workers, args.num_workers,
                                                                          args.device, args.distributed,
                                                                          epoch_changed_event_signal_slots)

    return data_loader_train, data_loader_val, epoch_changed_event_signal_slots


def build_training_actor_and_dataloader(args, network_config: dict, train_config: dict, train_dataset_config_path: str,
                                        val_dataset_config_path: str):
    data_loader_train, data_loader_val, epoch_changed_event_signal_slots = _build_dataloader(args, network_config,
                                                                                             train_config,
                                                                                             train_dataset_config_path,
                                                                                             val_dataset_config_path)

    actor = build_siam_encoder_detr_tracking_training_actor(args, network_config, train_config,
                                                            epoch_changed_event_signal_slots)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        actor.load_state_dict(checkpoint)
        args.start_epoch = actor.get_epoch()
    else:
        actor.reset_parameters()

    return actor, data_loader_train, data_loader_val
