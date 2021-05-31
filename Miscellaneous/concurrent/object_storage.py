import ctypes
import multiprocessing
import multiprocessing.sharedctypes
import pickle


class SharedMemory_ObjectStorage:
    def __init__(self, number_of_objects: int, bucket_size: int):
        self.number_of_objects = number_of_objects
        self.bucket_size = bucket_size
        self.shared_memory = multiprocessing.sharedctypes.RawArray(ctypes.c_char, number_of_objects * (bucket_size + 8))
        self.lock = multiprocessing.Lock()

    def save(self, object_, index):
        serialized_object = pickle.dumps(object_)
        serialized_object_size = len(serialized_object)
        assert serialized_object_size <= self.bucket_size, 'object too large for the bucket size'

        offset = index * (self.bucket_size + 8)

        with self.lock:
            self.shared_memory[offset: offset + 8] = serialized_object_size.to_bytes(4, byteorder='little', signed=False)
            self.shared_memory[offset + 8: offset + 8 + serialized_object_size] = serialized_object

    def load(self, index):
        offset = index * (self.bucket_size + 8)

        with self.lock:
            serialized_object_size = int.from_bytes(self.shared_memory[offset: offset + 8], byteorder='little', signed=False)
            serialized_object = bytearray(self.shared_memory[offset + 8: offset + 8 + serialized_object_size])

        return pickle.loads(serialized_object)
