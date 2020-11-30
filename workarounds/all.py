def apply_all_workarounds():
    from .numpy import numpy_no_multithreading
    from .opencv import opencv_no_multithreading
    from .reproducibility import seed_all_rng, enable_deterministic_computation

    numpy_no_multithreading()
    opencv_no_multithreading()
    seed_all_rng()
    enable_deterministic_computation()
