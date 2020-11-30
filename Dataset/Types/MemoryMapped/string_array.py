import os
import codecs
import numpy as np


def _str_to_np_array(string: str):
    raw_bytes = string.encode('utf-8')
    return np.frombuffer(raw_bytes, dtype=np.uint8)


def _np_array_to_str(array: np.ndarray):
    decoder = codecs.getdecoder("utf-8")
    return decoder(array)[0]


class StringArrayMemoryMapped:
    def __init__(self, path, indices):
        assert indices.shape[0] != 0
        self.strings = np.memmap(path, mode='readonly', shape=(indices[indices.shape[0] - 1],))
        self.indices = indices

    def __len__(self):
        return self.indices.shape[0]

    def __getitem__(self, index: int):
        if index == 0:
            begin_index = 0
        else:
            begin_index = self.indices[index - 1]
        end_index = self.indices[index]
        string = self.strings[begin_index: end_index]
        return _np_array_to_str(string)


class StringArrayMemoryMappedConstructor:
    def __init__(self, path):
        self.path = path
        self.strings = []
        self.string_indices = []
        self.current_index = 0

    def add(self, string: str):
        array = _str_to_np_array(string)
        self.strings.append(array)
        self.current_index += array.shape[0]
        self.string_indices.append(self.current_index)

    def construct(self):
        raw_string = np.concatenate(self.strings)
        path = self.path + '.lock'
        if os.path.exists(path):
            os.remove(path)

        strings = np.memmap(path, mode='write', shape=raw_string.shape)
        strings[:] = raw_string[:]
        del strings
        if os.path.exists(self.path):
            os.remove(self.path)
        os.rename(path, self.path)
        return StringArrayMemoryMapped(self.path, np.array(self.string_indices))
