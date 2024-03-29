def apply_all_workarounds(seed=0):
    from .numpy import numpy_no_multithreading
    # from .opencv import opencv_no_multithreading
    from .reproducibility import seed_all_rng, enable_deterministic_computation
    # from .Tensorflow import silence_tensorflow, disable_tensorflow_gpu
    # from .torchvision_tensorflow import fix_torchvision_tensorflow

    # fix_torchvision_tensorflow()
    numpy_no_multithreading()
    # opencv_no_multithreading()
    seed_all_rng(seed)
    # enable_deterministic_computation()
    # silence_tensorflow()

    # disable_tensorflow_gpu()
