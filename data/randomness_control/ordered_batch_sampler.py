import torch
import torch.cuda
from torch.utils.data.sampler import Sampler
from torch.utils.data.dataloader import default_collate
import torch.utils.data._utils.pin_memory
from typing import List


class OrderedBatchSampler(Sampler[List[int]]):
    r"""Wraps another sampler to yield a mini-batch of indices.

    Args:
        sampler (Sampler or Iterable): Base sampler. Can be any iterable object
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``

    Example:
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=False))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """

    def __init__(self, sampler: Sampler[int], batch_size: int, collate_fn=default_collate, pin_memory=False) -> None:
        # Since collections.abc.Iterable does not check for `__getitem__`, which
        # is one way for an object to be an iterable, we don't do an `isinstance`
        # check here.
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        self.sampler = sampler
        self.batch_size = batch_size
        self.collate_fn = collate_fn
        if pin_memory and torch.cuda.is_available():
            self.pin_memory = True
        else:
            self.pin_memory = False

    def __iter__(self):
        batch = {}
        iterations = 0
        for idx, data in self.sampler:
            batch[idx] = data
            while True:
                current_batch_range = range(iterations * self.batch_size, (iterations + 1) * self.batch_size)
                if all(idx in batch for idx in current_batch_range):
                    current_batch = []
                    for idx in current_batch_range:
                        current_batch.append(batch.pop(idx))
                    collated_data = self.collate_fn(current_batch)
                    if self.pin_memory:
                        collated_data = torch.utils.data._utils.pin_memory.pin_memory(collated_data)
                    yield collated_data
                    iterations += 1
                else:
                    break

    def __len__(self):
        # Can only be called if self.sampler has __len__ implemented
        # We cannot enforce this condition, so we turn off typechecking for the
        # implementation below.
        # Somewhat related: see NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]
        return len(self.sampler) // self.batch_size  # type: ignore