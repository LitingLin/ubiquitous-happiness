import torch
from training.transt.runner import TransTRunner
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


def build_transt_training_runner(args, net_config: dict, train_config: dict, additional_state_objects, training_start_event_slots, training_stop_event_slots, epoch_changed_event_slots, statistics_collectors):
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

    return TransTRunner(model, criterion, optimizer, lr_scheduler, additional_state_objects, training_start_event_slots, training_stop_event_slots, epoch_changed_event_slots, statistics_collectors)


def build_training_actor_and_dataloader(args, network_config: dict, train_config: dict, train_dataset_config_path: str,
                                        val_dataset_config_path: str):
    stateful_objects, training_start_event_signal_slots, training_stop_event_signal_slots, statistics_collectors = (None, None, None, None)
    if 'dataloader' not in train_config or train_config['dataloader']['version'] == 'old':
        from .dataloader.old import build_old_dataloader
        (train_dataset, val_dataset),\
        (data_loader_train, data_loader_val),\
        epoch_changed_event_slots = build_old_dataloader(args,
                                                         network_config, train_config,
                                                         train_dataset_config_path, val_dataset_config_path)
    elif train_config['dataloader']['version'] == 1:
        from .dataloader.v1 import build_dataloader
        (train_dataset, val_dataset),\
        (data_loader_train, data_loader_val),\
        epoch_changed_event_slots = build_dataloader(args,
                                                     network_config, train_config,
                                                     train_dataset_config_path, val_dataset_config_path)
    elif train_config['dataloader']['version'] == 2:
        from .dataloader.v2 import build_dataloader
        (train_dataset, val_dataset),\
        (data_loader_train, data_loader_val),\
        (stateful_objects, training_start_event_signal_slots, training_stop_event_signal_slots, epoch_changed_event_slots, statistics_collectors)\
            = build_dataloader(args, network_config, train_config, train_dataset_config_path, val_dataset_config_path)
    else:
        raise NotImplementedError(f'Unknown dataloader version {train_config["dataloader"]["version"]}')

    runner = build_transt_training_runner(args, network_config, train_config, stateful_objects, training_start_event_signal_slots, training_stop_event_signal_slots, epoch_changed_event_slots, statistics_collectors)

    if args.resume:
        model_state_dict, training_state_dict = load_checkpoint(args.resume)
        runner.load_state_dict(model_state_dict, training_state_dict)
        args.start_epoch = runner.get_epoch()

    return runner, data_loader_train, data_loader_val
