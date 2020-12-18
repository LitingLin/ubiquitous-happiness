import torch
from .actor import MultiresSiamFCFrontEndDeformDETRTrackingTrainingActor
from models.network.deformable_detr_tracking.build_siamfc_multi_res_deform_atten_track import build_siamfc_multires_deform_atten_track, initialize_siamfc_multires_deform_atten_track
from models.loss.detr_tracking.builder import build_detr_tracking_loss
from data.siamfc.dataset import build_tracking_dataset
from data.siamfc.processor.z_curate_x_resize import SiamFCZCurateXResizeProcessor
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler


def build_multires_siamfc_frontend_deform_detr_tracking_training_actor(args, net_config: dict, train_config: dict, device, distributed_samplers):
    model = build_siamfc_multires_deform_atten_track(net_config)

    criterion = build_detr_tracking_loss(train_config)

    # lr_backbone_names = ["backbone.0", "backbone.neck", "input_proj", "transformer.encoder"]
    def match_name_keywords(n, name_keywords):
        out = False
        for b in name_keywords:
            if b in n:
                out = True
                break
        return out

    for n, p in model.named_parameters():
        print(n)

    lr_backbone_names = ["backbone"]
    lr_linear_proj_names = ['reference_points', 'sampling_offsets']

    param_dicts = [
        {
            "params":
                [p for n, p in model.named_parameters()
                 if not match_name_keywords(n, lr_backbone_names) and not match_name_keywords(n, lr_linear_proj_names) and p.requires_grad],
            "lr": train_config['train']['lr'],
        },
        {
            "params": [p for n, p in model.named_parameters() if match_name_keywords(n, lr_backbone_names) and p.requires_grad],
            "lr": train_config['train']['lr_backbone'],
        },
        {
            "params": [p for n, p in model.named_parameters() if match_name_keywords(n, lr_linear_proj_names) and p.requires_grad],
            "lr": train_config['train']['lr'] * train_config['train']['lr_linear_proj_mult'],
        }
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

    model.to(device)
    criterion.to(device)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    return MultiresSiamFCFrontEndDeformDETRTrackingTrainingActor(model, criterion, optimizer, lr_scheduler, distributed_samplers)


def _build_dataloader(args, network_config: dict, train_config: dict, train_dataset_config_path: str, val_dataset_config_path: str):
    processor = SiamFCZCurateXResizeProcessor(network_config['backbone']['siamfc']['exemplar_size'], network_config['backbone']['siamfc']['instance_size'], network_config['backbone']['siamfc']['context'])
    train_dataset, val_dataset = build_tracking_dataset(train_config, train_dataset_config_path, val_dataset_config_path, processor, processor)

    set_on_epoch_changed = []

    if args.distributed:
        sampler_train = DistributedSampler(train_dataset)
        sampler_val = DistributedSampler(val_dataset, shuffle=False)
        set_on_epoch_changed.extend([sampler_train, sampler_val])
    else:
        sampler_train = torch.utils.data.RandomSampler(train_dataset)
        sampler_val = torch.utils.data.SequentialSampler(val_dataset)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, train_config['train']['batch_size'], drop_last=True)

    data_loader_train = DataLoader(train_dataset, batch_sampler=batch_sampler_train,
                                   num_workers=args.num_workers)
    data_loader_val = DataLoader(val_dataset, train_config['val']['batch_size'], sampler=sampler_val,
                                 drop_last=False, num_workers=args.num_workers)
    return data_loader_train, data_loader_val, set_on_epoch_changed


def build_training_actor_and_dataloader(args, network_config: dict, train_config: dict, train_dataset_config_path: str, val_dataset_config_path: str):
    data_loader_train, data_loader_val, set_on_epoch_changed = _build_dataloader(args, network_config, train_config, train_dataset_config_path, val_dataset_config_path)
    device = torch.device(args.device)

    actor = build_multires_siamfc_frontend_deform_detr_tracking_training_actor(args, network_config, train_config, device, set_on_epoch_changed)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        actor.load_state_dict(checkpoint)
        args.start_epoch = actor.get_epoch()

    return actor, data_loader_train, data_loader_val
