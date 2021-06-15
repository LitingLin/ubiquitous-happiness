import numpy as np
import multiprocessing
import multiprocessing.sharedctypes
import ctypes


class RandomSamplerWithoutReplacement_SharedMemory:
    def __init__(self, length, initial_seed):
        assert length > 0
        self.shared_memory = multiprocessing.sharedctypes.RawArray(ctypes.c_long, length + 2)
        self.lock = multiprocessing.Lock()
        self._initialize_unsafe(length, initial_seed)

    def _initialize_unsafe(self, length, seed):
        np_array = np.ctypeslib.as_array(self.shared_memory)
        np_array[0] = seed
        np_array[1] = 0
        np_array[2:] = np.arange(length)

        rng_engine = np.random.Generator(np.random.PCG64(seed))
        rng_engine.shuffle(np_array[2:])

    def __len__(self):
        return len(self.shared_memory) - 2

    def get_next(self):
        with self.lock:
            index = self.shared_memory[1]
            if index == len(self):
                self._reset_unsafe()
                index = 0
            value = self.shared_memory[index + 2]
            self.shared_memory[1] += 1
        return value

    def _reset_unsafe(self):
        np_array = np.ctypeslib.as_array(self.shared_memory)
        np_array[0] += 1
        seed = np_array[0]
        rng_engine = np.random.Generator(np.random.PCG64(seed))
        rng_engine.shuffle(np_array[2:])
        np_array[1] = 0

    def reset(self):
        with self.lock:
            self._reset_unsafe()

    def __getstate__(self):
        # unsafe
        return np.ctypeslib.as_array(self.shared_memory).copy()

    def __setstate__(self, state):
        # unsafe
        np_array = np.ctypeslib.as_array(self.shared_memory)
        np_array[:] = state
