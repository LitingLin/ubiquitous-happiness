import multiprocessing.shared_memory
from Miscellaneous.torch.distributed import is_dist_available_and_initialized, get_rank


class RandomSamplerWithoutReplacement_DistributedAwareness:
    def __init__(self, length, seed):
        assert length > 0
        self.shared_memory = multiprocessing.sharedctypes.RawArray(ctypes.c_long, length + 2)
        self.lock = multiprocessing.Lock()
        self._initialize_unsafe(length, initial_seed)
        pass

    def _unsafe_initialize(self):
        pass


class RandomSamplerWithoutReplacement_SharedMemory:
    def __init__(self, length, seed):
        pass
