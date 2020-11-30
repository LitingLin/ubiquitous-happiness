from Dataset.DataSplit import DataSplit
from Dataset._base_seed import BaseSeed


class UAV_Benchmark_M_Seed(BaseSeed):
    def __init__(self, root_path: str=None, annotation_path: str = None, data_split: DataSplit = DataSplit.Training | DataSplit.Validation):
        if root_path is None:
            root_path = self._getPathFromConfig('UAV_Benchmark_M_PATH')
        super(UAV_Benchmark_M_Seed, self).__init__('UAV-Benchmark-M', root_path, data_split, 2)
        self.annotation_path = annotation_path

    def construct(self, constructor):
        from ..Sources.UAV_Benchmark_M import construct_UAV_Benchmark_M
        construct_UAV_Benchmark_M(constructor, self)
