import numpy as np
import imageio
from typing import Union
from enum import IntFlag, auto
import pickle
import lzma


class LosslessImage:
    class ImageType(IntFlag):
        GRAY = auto()
        RGB = auto()

    data: np.ndarray

    def read_image(self, path: str):

        self.data = imageio.imread(path)

    def read_from_ppm(self, file_path: str):
        def _read_ascii_digit(string: bytes, begin_index: int):
            zero_value = ord(b'0')
            nine_value = ord(b'9')
            end_index = begin_index
            while zero_value <= string[end_index] <= nine_value:
                end_index += 1
            return int(string[begin_index: end_index]), end_index

        def _is_whitespace(value: int):
            return value in [ord(b'\r'), ord(b'\n'), ord(b'\t'), ord(b' ')]

        def _ignore_whitespaces(string: bytes, begin_index: int):
            while _is_whitespace(string[begin_index]):
                begin_index += 1
            return begin_index

        with open(file_path, 'rb') as fid:
            file_content = fid.read()

        if file_content[0:2] == b'P5':
            image_type = LosslessImage.ImageType.GRAY
        elif file_content[0:2] == b'P6':
            image_type = LosslessImage.ImageType.RGB
        else:
            raise Exception("Unsupported file format")

        width_begin_index = _ignore_whitespaces(file_content, 2)
        assert width_begin_index != 2

        width, width_end_index = _read_ascii_digit(file_content, width_begin_index)
        height_begin_index = _ignore_whitespaces(file_content, width_end_index)
        assert height_begin_index != width_end_index
        height, height_end_index = _read_ascii_digit(file_content, height_begin_index)
        depth_begin_index = _ignore_whitespaces(file_content, height_end_index)
        depth, depth_end_index = _read_ascii_digit(file_content, depth_begin_index)
        assert 0 < depth < 65536
        assert _is_whitespace(file_content[depth_end_index])

        image_data = memoryview(file_content)[depth_end_index + 1:]
        if depth < 256:
            data = np.frombuffer(image_data, dtype=np.uint8)
            data = data.copy()
        else:
            np_data_type = np.dtype(np.uint16)
            np_data_type = np_data_type.newbyteorder('B')
            data = np.frombuffer(image_data, dtype=np_data_type)
            data = data.astype(np.uint16)

        if image_type == LosslessImage.ImageType.GRAY:
            data = data.reshape([height, width])
        elif image_type == LosslessImage.ImageType.RGB:
            data = data.reshape([height, width, 3])
        else:
            raise Exception

        self.data = data

    def serialize(self):
        return lzma.compress(pickle.dumps(self.__dict__))

    def deserialize(self, raw_data: Union[bytes, memoryview]):
        self.__dict__.update(pickle.loads(lzma.decompress(raw_data)))

    def getImage(self):
        return self.data
