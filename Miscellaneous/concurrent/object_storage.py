import ctypes
import multiprocessing
import multiprocessing.sharedctypes
import pickle
from Miscellaneous.Numpy.from_ctypes import make_nd_array
from Miscellaneous.Numpy.to_ctypes import memmove_to_ctypes
import numpy as np


class SharedMemory_ObjectStorage:
    def __init__(self, number_of_objects: int, bucket_size: int):
        self.number_of_objects = number_of_objects
        self.bucket_size = bucket_size
        self.shared_memory = multiprocessing.sharedctypes.RawArray(ctypes.c_char, number_of_objects * (bucket_size + 8))
        self.lock = multiprocessing.Lock()

    def save(self, index, object_):
        serialized_object = pickle.dumps(object_)
        serialized_object_size = len(serialized_object)
        assert serialized_object_size <= self.bucket_size, 'object too large for the bucket size'

        offset = index * (self.bucket_size + 8)

        with self.lock:
            self.shared_memory[offset: offset + 8] = serialized_object_size.to_bytes(8, byteorder='little', signed=False)
            self.shared_memory[offset + 8: offset + 8 + serialized_object_size] = serialized_object

    def load(self, index):
        offset = index * (self.bucket_size + 8)

        with self.lock:
            serialized_object_size = int.from_bytes(self.shared_memory[offset: offset + 8], byteorder='little', signed=False)
            serialized_object = bytes(self.shared_memory[offset + 8: offset + 8 + serialized_object_size])

        return pickle.loads(serialized_object)

    def get_state(self):
        return make_nd_array(ctypes.addressof(self.shared_memory), (self.number_of_objects, self.bucket_size + 8), np.byte)

    def load_state(self, state: np.ndarray):
        memmove_to_ctypes(state, self.shared_memory)
