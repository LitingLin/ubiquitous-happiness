import numpy as np


def positive_negative_samples_controlling(positive_samples, total_samples, target_positive_samples_ratio, rng_engine=np.random):
    if target_positive_samples_ratio == 0:
        return -1
    if target_positive_samples_ratio == 1:
        return 1

    current_positive_ratio = positive_samples / total_samples
    positive_ratio = target_positive_samples_ratio - (current_positive_ratio - target_positive_samples_ratio) * 0.5
    is_positive = rng_engine.rand() < positive_ratio
    if abs(current_positive_ratio - target_positive_samples_ratio) < 0.05:
        return 0 if is_positive else -1
    else:
        return 1 if is_positive else -1
