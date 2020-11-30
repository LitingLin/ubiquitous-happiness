import numpy as np
import codecs


def _str_to_np_array(string: str):
    raw_bytes = string.encode('utf-8')
    return np.frombuffer(raw_bytes, dtype=np.uint8)


def _np_array_to_str(array: np.ndarray):
    decoder = codecs.getdecoder("utf-8")
    return decoder(array)[0]


class _BatchWriter:
    def __init__(self, parent):
        self.parent = parent

    def append(self, string: str):
        array = _str_to_np_array(string)
        shape = array.shape[0]
        self.to_write.append(array)
        self.current_index += shape
        self.indices.append(self.current_index)

    def __enter__(self):
        self.to_write = [self.parent.encoded]
        self.indices = []
        if self.parent.indices.shape[0] > 0:
            self.current_index = self.parent.indices[self.parent.indices.shape[0] - 1]
        else:
            self.current_index = 0
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if len(self.indices) > 0:
            self.parent.encoded = np.concatenate(self.to_write)
            self.parent.indices = np.append(self.parent.indices, self.indices)


class StringArrayNumpy:
    encoded: np.ndarray
    indices: np.ndarray

    def __init__(self):
        self.encoded = np.empty([0], dtype=np.uint8)
        self.indices = np.empty([0], dtype=np.uint32)

    def append(self, string: str):
        pass

    def remove(self, index: int):
        pass

    def __len__(self):
        return self.indices.shape[0]

    def __getitem__(self, index: int):
        if index == 0:
            begin_index = 0
        else:
            begin_index = self.indices[index - 1]
        end_index = self.indices[index]
        string = self.encoded[begin_index: end_index]
        return _np_array_to_str(string)

    def __setitem__(self, key, value):
        pass

    def batchWriter(self):
        return _BatchWriter(self)


if __name__ == '__main__':
    a = StringArrayNumpy()
    with a.batchWriter() as writer:
        writer.append('sfdsf')
        writer.append('dsfdsf')
    print(a[0])
    print(a[1])
    assert len(a) == 2
