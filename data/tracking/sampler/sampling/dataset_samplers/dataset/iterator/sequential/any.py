from data.tracking.sampler.sampling.dataset_samplers._sampling_algos.stateful.sequential import Sampling_SequentialIndexing
from data.tracking.sampler.sampling.dataset_samplers.dataset.iterator._template import DatasetSamplerTemplate


class DatasetSampler_Sequential(DatasetSamplerTemplate):
    def __init__(self, dataset):
        super(DatasetSampler_Sequential, self).__init__(Sampling_SequentialIndexing(len(dataset)), dataset)
