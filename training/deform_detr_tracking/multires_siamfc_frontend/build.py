import torch
from ..actor import DETRTrackingActor
from models.network.deformable_detr_tracking.build_siamfc_multi_res_deform_atten_track import build_siamfc_multires_deform_atten_track, initialize_siamfc_multires_deform_atten_track
from models.loss.detr_tracking.builder import build_detr_tracking_loss
from data.siamfc.dataset import TrackingDataset
from data.siamfc.detr_tracking_processor import DETRTrackingProcessor


class MultiresSiamFCFrontEndDeformDETRTrackingTrainingActor(DETRTrackingActor):
    def reset_parameters(self):
        initialize_siamfc_multires_deform_atten_track(self.get_model())


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


def build_dataloader(train_config: dict, eval_config: dict):
    TrackingDataset(train_config, )


def resume_training_actor_and_dataloader():
    pass


def build_training_actor_and_dataloader():
    pass
