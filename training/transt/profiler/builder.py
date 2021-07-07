def build_profiler(args):
    if args.enable_profile:
        from .pytroch import PytorchProfiler
        return PytorchProfiler(args.profile_logging_path, args.device)
    else:
        from .dummy import DummyProfiler
        return DummyProfiler()


def build_efficiency_assessor(model, pseudo_data_generator, train_config):
    batch = train_config['data']['sampler']['train']['batch_size']
    from .efficiency_assessor import TrackerEfficiencyAssessor
    return TrackerEfficiencyAssessor(model, pseudo_data_generator, batch)
