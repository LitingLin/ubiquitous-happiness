from ._common import _BaseFilter
import numpy as np


class SortByImageRatio(_BaseFilter):
    def __call__(self, sizes):
        sizes = np.array(sizes)
        ratio = sizes[:, 1] / sizes[:, 0]
        indices = ratio.argsort()
        return indices
