from data.tracking.sampler.sampling.dataset_samplers._indexing.random_without_replacement import RandomSamplingWithoutReplacement
from data.tracking.sampler.sampling.dataset_samplers.dataset.iterator._template import DatasetSamplerTemplate


class DatasetSampler_RandomWithoutReplacement(DatasetSamplerTemplate):
    def __init__(self, dataset, rng_engine):
        super(DatasetSampler_RandomWithoutReplacement, self).__init__(RandomSamplingWithoutReplacement(len(dataset), rng_engine), dataset)
