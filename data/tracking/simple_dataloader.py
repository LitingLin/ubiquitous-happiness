from Miscellaneous.simple_prefetcher import SimplePrefetcher
from native_extension import ImageDecoder


class _SimpleTrackingSequenceIterator:
    def __init__(self, sequence):
        self.sequence = sequence
        self.decoder = ImageDecoder()

    def __len__(self):
        return len(self.sequence)

    def __getitem__(self, index: int):
        if index >= len(self):
            raise IndexError

        image_path, bbox = self.sequence[index]
        image = self.decoder.decode(image_path)
        return image, bbox

    def __getattr__(self, item):
        return getattr(self.sequence, item)


class SimpleTrackingDatasetIterator:
    def __init__(self, dataset, prefetch=True):
        self.dataset = dataset
        self.prefetch = prefetch

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
        if index >= len(self):
            raise IndexError

        sequence_iterator = _SimpleTrackingSequenceIterator(self.dataset[index])
        if self.prefetch:
            return SimplePrefetcher(sequence_iterator)
        else:
            return sequence_iterator

    def __getattr__(self, item):
        return getattr(self.dataset, item)
