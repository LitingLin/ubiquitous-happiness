import multiprocessing
import multiprocessing.sharedctypes
import ctypes
import time

from Miscellaneous.type_limits import c_type_limits
ulonglong_max = c_type_limits(ctypes.c_ulonglong)[1]


class ResourceSchedulerContext:
    def __init__(self, index, used_count):
        self.index = index
        self.used_count = used_count

    def get_index(self):
        return self.index

    def get_used_count(self):
        return self.used_count


class ResourceSchedulerGuard:
    def __init__(self, scheduler):
        self.scheduler = scheduler

    def __enter__(self):
        self.context = self.scheduler.acquire()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.scheduler.release(self.context)


def _default_scheduling_constraint(allocation_state_list):
    return tuple(index for index in range(len(allocation_state_list)) if not allocation_state_list[index])


def _default_scheduling_preference(available_resources_indices, resources_used_times_counter_list):
    count = ulonglong_max
    index_of_resource_acquired = None
    for index in available_resources_indices:
        if resources_used_times_counter_list[index] < count:
            index_of_resource_acquired = index
            count = resources_used_times_counter_list[index]
    return index_of_resource_acquired


class ResourceScheduler_MultiProcessingAwareness:
    def __init__(self, number_of_resources, scheduling_constraint_fn=None, scheduling_preference_fn=_default_scheduling_preference):
        self.number_of_resources = number_of_resources
        self.lock = multiprocessing.Lock()
        self.counter = multiprocessing.sharedctypes.RawArray(ctypes.c_ulonglong, number_of_resources)
        self.allocation = multiprocessing.sharedctypes.RawArray(ctypes.c_bool, number_of_resources)
        self.scheduling_constraint_fn = scheduling_constraint_fn
        self.scheduling_preference_fn = scheduling_preference_fn

    def set_scheduling_constraint_function(self, constraint_fn):
        self.scheduling_constraint_fn = constraint_fn

    def _try_acquire_unsafe(self):
        available_resources_indices = tuple(index for index in range(self.number_of_resources) if not self.allocation[index])
        if len(available_resources_indices) == 0:
            return False, None, None
        if self.scheduling_constraint_fn is not None:
            available_resources_indices = self.scheduling_constraint_fn(available_resources_indices, self.counter)
            if len(available_resources_indices) == 0:
                return False, None, None

        index_of_resource_acquired = self.scheduling_preference_fn(available_resources_indices, self.counter)

        self.allocation[index_of_resource_acquired] = True
        return True, index_of_resource_acquired, self.counter[index_of_resource_acquired]

    def acquire(self, retry_interval=0.01, timeout=0):
        if timeout != 0:
            begin_time = time.perf_counter()
        while True:
            with self.lock:
                acquired, index_of_resource, resource_count = self._try_acquire_unsafe()
            if acquired:
                return ResourceSchedulerContext(index_of_resource, resource_count)
            if timeout != 0:
                if time.perf_counter() - begin_time > timeout:
                    raise TimeoutError
            time.sleep(retry_interval)

    def _release_unsafe(self, index):
        assert self.allocation[index]
        self.counter[index] += 1
        self.allocation[index] = False

    def release(self, context: ResourceSchedulerContext):
        with self.lock:
            self._release_unsafe(context.index)

    def acquire_and_release(self, context_to_release=None, retry_interval=0.01, timeout=0):
        with self.lock:
            if context_to_release is not None:
                self._release_unsafe(context_to_release)
            acquired, index_of_resource, resource_count = self._try_acquire_unsafe()
        if acquired:
            return ResourceSchedulerContext(index_of_resource, resource_count)
        return self.acquire(retry_interval, timeout)

    def reset_counter(self):
        with self.lock:
            ctypes.memset(ctypes.addressof(self.counter), 0, ctypes.sizeof(self.counter))
