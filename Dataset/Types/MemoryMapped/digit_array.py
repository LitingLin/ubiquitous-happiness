import os
import numpy as np


class DigitMatrixMemoryMapped:
    matrix: np.ndarray
    def __init__(self, path, dtype, shape):
        self.matrix = np.memmap(path, dtype=dtype, mode='readonly', shape=shape)

    def __getitem__(self, item):
        return self.matrix[item]

class DigitMatrixMemoryMappedConstructor:
    def __init__(self, path):
        self.path = path
        self.cache = []

    def add(self, value):
        self.cache.append(value)

    def construct(self):
        np_matrix = np.array(self.cache)
        path = self.path + '.lock'
        if os.path.exists(path):
            os.remove(path)

        memory_mapped_matrix = np.memmap(path, dtype=np_matrix.dtype, mode='write', shape=np_matrix.shape)
        memory_mapped_matrix[:] = np_matrix[:]
        del memory_mapped_matrix
        if os.path.exists(self.path):
            os.remove(self.path)
        os.rename(path, self.path)
        return DigitMatrixMemoryMapped(self.path, np_matrix.dtype, np_matrix.shape)
