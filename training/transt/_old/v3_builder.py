from training.transt.runner import TransTRunner
from models.TransT.builder import build_transt
from models.TransT.loss._old.v3_builder import build_criterion
from miscellanies.torch.checkpoint import load_checkpoint
from data.tracking.methods.TransT.training.builder import build_stage_2_data_processor
from training.transt.logger.builder import build_logger
from training.transt.profiler.builder import build_profiler, build_efficiency_assessor
from data.tracking.methods.TransT.pseudo_data import build_pseudo_data_generator
import torch
import torch.distributed


def _get_n_epochs(train_config: dict):
    if 'version' not in train_config or train_config['version'] < 3:
        epochs = train_config['train']['epochs']
    else:
        epochs = train_config['optimization']['epochs']
    return epochs


def _get_clip_max_norm(train_config: dict):
    if 'version' not in train_config or train_config['version'] < 3:
        clip_max_norm = train_config['train']['clip_max_norm']
    else:
        clip_max_norm = None
        if 'clip_max_norm' in train_config['optimization']:
            clip_max_norm = train_config['optimization']['clip_max_norm']
    return clip_max_norm


def setup_optimizer(model, network_config: dict, train_config: dict):
    from ..optimization.transt import build_transt_optimizer
    return build_transt_optimizer(model, train_config)


def build_transt_training_runner(args, net_config: dict, train_config: dict,
                                 stage_2_data_processor,
                                 additional_state_objects,
                                 training_start_event_slots, training_stop_event_slots,
                                 epoch_changed_event_slots, statistics_collectors):
    model = build_transt(net_config, True)
    device = torch.device(args.device)

    criterion = build_criterion(net_config, train_config)
    optimizer, lr_scheduler = setup_optimizer(model, net_config, train_config)

    if hasattr(criterion, 'set_epoch'):
        epoch_changed_event_slots.append(criterion)

    if 'sync_bn' in train_config['optimization'] and 'cuda' in device.type:
        if train_config['optimization']['sync_bn']:
            from miscellanies.torch.distributed import is_dist_available_and_initialized
            if is_dist_available_and_initialized():
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model.to(device)
    criterion.to(device)

    if args.distributed:
        if 'cuda' in args.device:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model = torch.nn.parallel.DistributedDataParallel(model)

            check_dist_model_params_equal = False
            if check_dist_model_params_equal:
                from miscellanies.torch.distributed import get_rank, get_world_size
                gathered_objects = [None for _ in range(get_world_size())]
                torch.distributed.all_gather_object(gathered_objects, dict(model.module.named_parameters()))
                for i in range(1, get_world_size()):
                    for v1, v2 in zip(gathered_objects[0].values(), gathered_objects[1].values()):
                        assert torch.equal(v1, v2)
                del gathered_objects

    grad_max_norm = _get_clip_max_norm(train_config)

    return TransTRunner(model, criterion, optimizer, lr_scheduler,
                        grad_max_norm,
                        stage_2_data_processor,
                        additional_state_objects,
                        training_start_event_slots, training_stop_event_slots,
                        epoch_changed_event_slots, statistics_collectors)


def build_training_dataloader(args, network_config: dict, train_config: dict, train_dataset_config_path: str,
                                         val_dataset_config_path: str):
    if 'version' not in train_config or train_config['version'] < 3:
        import training.transt._old.builder
        return training.transt._old.builder.build_training_actor_and_dataloader(args, network_config, train_config, train_dataset_config_path, val_dataset_config_path)
    stateful_objects, training_start_event_signal_slots, training_stop_event_signal_slots, statistics_collectors = (None, None, None, None)

    sampler_version = train_config['data']['sampler']['version']

    if sampler_version == 'old':
        from training.transt.dataloader.old import build_old_dataloader
        (train_dataset, val_dataset),\
        (data_loader_train, data_loader_val),\
        epoch_changed_event_slots = build_old_dataloader(args,
                                                         network_config, train_config,
                                                         train_dataset_config_path, val_dataset_config_path)
    elif sampler_version == 1:
        raise NotImplementedError('Deprecated')
        # from training.transt.dataloader.v1 import build_dataloader
        # (train_dataset, val_dataset),\
        # (data_loader_train, data_loader_val),\
        # epoch_changed_event_slots = build_dataloader(args,
        #                                              network_config, train_config,
        #                                              train_dataset_config_path, val_dataset_config_path)
    elif sampler_version == 2:
        from training.transt.dataloader.v2 import build_dataloader
        (train_dataset, val_dataset),\
        (data_loader_train, data_loader_val),\
        (stateful_objects, training_start_event_signal_slots, training_stop_event_signal_slots, epoch_changed_event_slots, statistics_collectors)\
            = build_dataloader(args, network_config, train_config, train_dataset_config_path, val_dataset_config_path)
    else:
        raise NotImplementedError(f'Unknown dataloader version {train_config["dataloader"]["version"]}')

    stage_2_data_processor = build_stage_2_data_processor(network_config, train_config)
    return (train_dataset, val_dataset),\
        (data_loader_train, data_loader_val),\
        (stateful_objects, training_start_event_signal_slots, training_stop_event_signal_slots, epoch_changed_event_slots, statistics_collectors), \
        stage_2_data_processor


def build_training_runner_logger_and_dataloader(args, network_config: dict, train_config: dict, train_dataset_config_path: str,
                                         val_dataset_config_path: str):
    (train_dataset, val_dataset), \
    (data_loader_train, data_loader_val), \
    (stateful_objects, training_start_event_signal_slots, training_stop_event_signal_slots, epoch_changed_event_slots,
     statistics_collectors), \
    stage_2_data_processor = build_training_dataloader(args, network_config, train_config, train_dataset_config_path, val_dataset_config_path)
    runner = build_transt_training_runner(args, network_config, train_config, stage_2_data_processor, stateful_objects, training_start_event_signal_slots, training_stop_event_signal_slots, epoch_changed_event_slots, statistics_collectors)

    n_epochs = _get_n_epochs(train_config)

    if args.resume:
        model_state_dict, training_state_dict = load_checkpoint(args.resume)
        runner.load_state_dict(model_state_dict, training_state_dict)

    begin_step = len(data_loader_train) * runner.get_epoch()

    logger = build_logger(args, network_config, train_config, begin_step)
    profiler = build_profiler(args)

    pseudo_data_generator = build_pseudo_data_generator(args, network_config)
    efficiency_assessor = build_efficiency_assessor(runner.get_model(), pseudo_data_generator, train_config)

    return n_epochs, runner, logger, profiler, data_loader_train, data_loader_val, efficiency_assessor
