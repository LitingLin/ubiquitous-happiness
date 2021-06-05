import torch
from .runner import TransTRunner
from data.tracking.methods.TransT.training.builder import build_transt_data_processor
from data.tracking.builder.stateful.siamfc.data_loader import build_siamfc_sampling_dataloader
from models.TransT.builder import build_transt
from models.TransT.loss.builder import build_criterion
from Miscellaneous.torch.checkpoint import load_checkpoint


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


def build_transt_training_runner(args, net_config: dict, train_config: dict, data_loader_train, data_loader_val):
    model = build_transt(net_config, True)
    device = torch.device(args.device)

    criterion = build_criterion(train_config)
    optimizer, lr_scheduler = _setup_optimizer(model, train_config)

    model.to(device)
    criterion.to(device)

    if args.distributed:
        if 'cuda' in args.device:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model = torch.nn.parallel.DistributedDataParallel(model)

    return TransTRunner(model, criterion, optimizer, lr_scheduler, data_loader_train, data_loader_val)


def _build_dataloader(args, network_config: dict, train_config: dict, train_dataset_config_path: str,
                      val_dataset_config_path: str):
    processor, collate_fn = build_transt_data_processor(network_config, train_config)
    return build_siamfc_sampling_dataloader(args, train_config, train_dataset_config_path, val_dataset_config_path, processor, processor, args.seed, collate_fn)


def build_training_actor_and_dataloader(args, network_config: dict, train_config: dict, train_dataset_config_path: str,
                                        val_dataset_config_path: str):
    (data_loader_train, data_loader_val), (torch_data_loader_train, torch_data_loader_val) = _build_dataloader(args, network_config, train_config, train_dataset_config_path, val_dataset_config_path)

    runner = build_transt_training_runner(args, network_config, train_config, data_loader_train, data_loader_val)

    if args.resume:
        model_state_dict, training_state_dict = load_checkpoint(args.resume)
        runner.load_state_dict(model_state_dict, training_state_dict)
        args.start_epoch = runner.get_epoch()

    return runner, torch_data_loader_train, torch_data_loader_val