import numpy as np
from scipy.stats import truncnorm


def get_truncated_normal_generator(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


def truncated_normal_dist_generate_random_number(mean:float=0, sd:float=1, low:float=0, upp:float=10, rng_engine: np.random.Generator = np.random.default_rng(), size=None):
    return truncnorm.rvs(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd, random_state=rng_engine, size=size)


if __name__ == '__main__':
    import time
    start = time.perf_counter()
    rng_engine = np.random.default_rng()
    for _ in range(100):
        print(truncated_normal_dist_generate_random_number(mean=0, sd=0.2, low=0.1, upp=0.9, rng_engine=rng_engine))
    print(time.perf_counter() - start)
