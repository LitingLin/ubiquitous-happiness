import torch.utils.data
import numpy as np
from Miscellaneous.concurrent.object_storage import SharedMemory_ObjectStorage
from Miscellaneous.concurrent.resource_scheduler import ResourceScheduler_MultiProcessingAwareness


class IterableDatasetOrchestratorWorkerIterator:
    def __init__(self):
        pass

    def __next__(self):
        pass


class _WorkerSchedulerConstraint:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.position = 0

    def set_position(self, position):
        self.position = position

    def __call__(self, available_indices, counter):
        valid_indices = []
        for index in available_indices:
            current_element_position = counter[index] * self.batch_size + index
            if current_element_position >= self.position:
                valid_indices.append(index)

        return valid_indices


class IterableDatasetOrchestrator(torch.utils.data.dataset.IterableDataset):
    def __init__(self, randomness_control_awareness_iterable_dataset, batch_size, seed: np.random.SeedSequence):
        self.global_rng = np.random.Generator(np.random.PCG64(seed.spawn(1)[0]))
        self.batch_rng_storage = SharedMemory_ObjectStorage(batch_size, 1024)
        self.batch_scheduler = ResourceScheduler_MultiProcessingAwareness(batch_size, )
        self.iterator_dataset = randomness_control_awareness_iterable_dataset
        self.batch_size = batch_size

    def _initialize_batch_rng_storage(self, seed: np.random.SeedSequence):
        for index, child_seed in enumerate(seed.spawn(self.batch_size)):
            new_rng = np.random.Generator(np.random.PCG64(child_seed))
            self.batch_rng_storage.save(index, new_rng.__getstate__())

    def get_state(self):
        return self.global_rng.__getstate__(), self.batch_rng_storage.get_state()

    def load_state(self, state):
        self.global_rng.__setstate__(state[0])
        self.batch_rng_storage.load_state(state[1])

    @staticmethod
    def create_from_state():
        pass

    def __iter__(self):
        pass

    def __len__(self):
        pass

