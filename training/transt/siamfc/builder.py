def build_siamfc_training_actor_and_dataloader(args, network_config: dict, train_config: dict, train_dataset_config_path: str,
                                        val_dataset_config_path: str):
    stateful_objects, training_start_event_signal_slots, training_stop_event_signal_slots = (None, None, None)
    sampler_parameters = train_config['data']['sampler']
    if sampler_parameters['version'] == 'old':
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
        (stateful_objects, training_start_event_signal_slots, training_stop_event_signal_slots, epoch_changed_event_slots)\
            = build_dataloader(args, network_config, train_config, train_dataset_config_path, val_dataset_config_path)
    else:
        raise NotImplementedError(f'Unknown dataloader version {train_config["dataloader"]["version"]}')

    runner = build_transt_training_runner(args, network_config, train_config, stateful_objects, training_start_event_signal_slots, training_stop_event_signal_slots, epoch_changed_event_slots)

    if args.resume:
        model_state_dict, training_state_dict = load_checkpoint(args.resume)
        runner.load_state_dict(model_state_dict, training_state_dict)
        args.start_epoch = runner.get_epoch()

    return runner, data_loader_train, data_loader_val
