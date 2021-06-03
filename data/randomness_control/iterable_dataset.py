import torch
import torch.distributed
import torch.utils.data
import numpy as np
from Miscellaneous.concurrent.object_storage import SharedMemory_ObjectStorage
from Miscellaneous.concurrent.resource_scheduler import ResourceScheduler_MultiProcessingAwareness, ResourceSchedulerGuard
from Miscellaneous.torch.distributed import is_dist_available_and_initialized


class _WorkerSchedulerConstraint:
    def __init__(self):
        self.position = 0

    def set_position(self, position):
        self.position = position

    def __call__(self, available_indices, counter):
        valid_indices = []
        batch_size = len(counter)
        for index in available_indices:
            current_element_position = counter[index] * batch_size + index
            if current_element_position >= self.position:
                valid_indices.append(index)

        return valid_indices


class IterableDatasetOrchestratorWorkerIterator:
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.initial_position_offset = 0 if self.orchestrator.iterable_dataset.get_position() < 0 else self.orchestrator.iterable_dataset.get_position()
        self.worker_scheduler_constraint = _WorkerSchedulerConstraint()
        self.orchestrator.batch_scheduler.set_scheduling_constraint_function(self.worker_scheduler_constraint)
        self.dataset_length = len(self.orchestrator) * self.orchestrator.world_size
        self.world_batch_size = self.orchestrator.batch_size * self.orchestrator.world_size
        self.world_batch_offset = self.orchestrator.rank_id * self.orchestrator.batch_size
        self.local_rng = np.random.Generator(np.random.PCG64())

    def __next__(self):
        with ResourceSchedulerGuard(self.orchestrator.batch_scheduler) as task_context:
            position = self.world_batch_size * task_context.get_used_count() + self.world_batch_offset + task_context.get_index()
            if position >= self.dataset_length:
                raise StopIteration

            local_position = self.orchestrator.batch_size * task_context.get_used_count() + task_context.get_index()

            self.orchestrator.iterable_dataset.forward_to(self.initial_position_offset + position, self.orchestrator.global_rng)

            batch_element_rng_state = self.orchestrator.batch_rng_storage.load(task_context.get_index())
            self.local_rng.__setstate__(batch_element_rng_state)
            data = self.orchestrator.iterable_dataset.do_sampling(self.local_rng)
            self.worker_scheduler_constraint.set_position(local_position)
            self.orchestrator.batch_rng_storage.save(task_context.get_index(), self.local_rng.__getstate__())
            return local_position, data


class IterableDatasetOrchestrator(torch.utils.data.dataset.IterableDataset):
    def __init__(self, randomness_control_awareness_iterable_dataset, batch_size, rank_id, world_size, seed: np.random.SeedSequence, drop_last=True):
        assert drop_last, 'Pytorch distributed parallel does not support different size batch'
        self.global_rng = np.random.Generator(np.random.PCG64(seed.spawn(1)[0]))
        self.batch_rng_storage = SharedMemory_ObjectStorage(batch_size, 1024)
        self.batch_scheduler = ResourceScheduler_MultiProcessingAwareness(batch_size)
        self.iterable_dataset = randomness_control_awareness_iterable_dataset
        self.batch_size = batch_size

        self.rank_id = rank_id
        self.world_size = world_size
        self._initialize_batch_rng_storage(seed)
        self.drop_last = drop_last

    def _get_length(self):
        return len(self.iterable_dataset) // (self.world_size * self.batch_size) * self.batch_size

    def _initialize_batch_rng_storage(self, seed: np.random.SeedSequence):
        for index, child_seed in enumerate(seed.spawn(self.batch_size)):
            new_rng = np.random.Generator(np.random.PCG64(child_seed))
            self.batch_rng_storage.save(index, new_rng.__getstate__())

    def synchronize(self, target_epoch):
        epoch_length = self._get_length() * self.world_size
        target_position = target_epoch * epoch_length
        self.iterable_dataset.forward_to(target_position, self.global_rng)
        self.batch_scheduler.reset_counter()

    def get_state(self):
        batch_rng_storage_state = self.batch_rng_storage.get_state()
        if is_dist_available_and_initialized() and self.world_size > 1:
            all_batch_rng_storage_state = [None for _ in range(self.world_size)]
            torch.distributed.all_gather_object(all_batch_rng_storage_state, batch_rng_storage_state)
            batch_rng_storage_state = np.concatenate(all_batch_rng_storage_state)
        else:
            assert self.world_size == 1

        return self.global_rng.__getstate__(), batch_rng_storage_state, self.iterable_dataset.get_state()

    def load_state(self, state):
        global_rng_state, batch_rng_storage_state, iterable_dataset_state = state
        assert batch_rng_storage_state.shape[0] == self.world_size * self.batch_size

        self.global_rng.__setstate__(global_rng_state)
        self.batch_rng_storage.load_state(batch_rng_storage_state[self.rank_id * self.batch_size: (self.rank_id + 1) * self.batch_size, :])
        self.iterable_dataset.load_state(iterable_dataset_state)

    def __iter__(self):
        return IterableDatasetOrchestratorWorkerIterator(self)

    def __len__(self):
        return self._get_length()
