from Dataset._base_seed import BaseSeed
from Dataset.DataSplit import DataSplit


class UAV_Benchmark_S_Seed(BaseSeed):
    def __init__(self, root_path: str=None, annotation_path: str = None):
        if root_path is None:
            root_path = self._getPathFromConfig('UAVBenchmarkS_PATH')
        super(UAV_Benchmark_S_Seed, self).__init__('UAV-benchmark-S', root_path, DataSplit.Full, 1)
        self.annotation_path = annotation_path

    def construct(self, constructor):
        from ..Sources.UAV_Benchmark_S import construct_UAVBenchmarkS
        construct_UAVBenchmarkS(constructor, self)
