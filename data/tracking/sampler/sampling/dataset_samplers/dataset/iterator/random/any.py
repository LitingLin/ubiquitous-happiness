from data.tracking.sampler.sampling.dataset_samplers._sampling.random_without_replacement import Sampling_RandomSamplingWithoutReplacement
from data.tracking.sampler.sampling.dataset_samplers.dataset.iterator._template import DatasetSamplerTemplate


class DatasetSampler_RandomWithoutReplacement(DatasetSamplerTemplate):
    def __init__(self, dataset, rng_engine):
        seed = rng_engine.randint(-1024, 1024)
        super(DatasetSampler_RandomWithoutReplacement, self).__init__(Sampling_RandomSamplingWithoutReplacement(len(dataset), seed), dataset)
