def build_profiler(args):
    if args.enable_profile:
        from .pytroch import PytorchProfiler
        return PytorchProfiler(args.profile_logging_path, args.device)
    else:
        from .dummy import DummyProfiler
        return DummyProfiler()
