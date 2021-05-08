from data.tracking.sampler.sampling.dataset_samplers._indexing.sequential import SequentialIndexing
from data.tracking.sampler.sampling.dataset_samplers.dataset.iterator._template import DatasetSamplerTemplate


class DatasetSampler_Sequential(DatasetSamplerTemplate):
    def __init__(self, dataset):
        super(DatasetSampler_Sequential, self).__init__(SequentialIndexing(len(dataset)), dataset)
