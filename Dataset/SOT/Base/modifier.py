from .dataset import SingleObjectTrackingDataset
from ._impl import _set_image_path
import pathlib


class _Context:
    root_path: pathlib.Path
    has_object_category: bool
    has_fps: bool


class SingleObjectDatasetFrameModifier:
    def __init__(self, sequence, index: int, context: _Context, iterator=None):
        self.sequence = sequence
        self.index = index
        self.frame = sequence.frames[index]
        self.context = context
        self.parent_iterator = iterator

    def setBoundingBox(self, bounding_box):
        self.frame.bounding_box = bounding_box

    def setImagePath(self, path: str):
        self.frame.size, self.frame.image_path = _set_image_path(self.context.root_path, path)

    def setAttributeIsPresent(self, is_present: bool):
        self.frame.is_present = is_present

    def getBoundingBox(self):
        if hasattr(self.frame, 'bounding_box'):
            return self.frame.bounding_box
        return None

    def getAttributeIsPresent(self):
        if hasattr(self.frame, 'is_present'):
            return self.frame.is_present
        return None

    def getImageSize(self):
        return self.frame.size

    def delete(self):
        self.sequence.frames.pop(self.index)
        if self.parent_iterator is not None:
            self.parent_iterator.index -= 1
        del self


class SingleObjectDatasetFrameIterator:
    def __init__(self, sequence, context):
        self.sequence = sequence
        self.context = context
        self.index = 0

    def __next__(self):
        if self.index >= len(self.sequence.frames):
            raise StopIteration

        frame_modifier = SingleObjectDatasetFrameModifier(self.sequence, self.index, self.context, self)
        self.index += 1
        return frame_modifier


class SingleObjectDatasetSequenceModifier:
    def __init__(self, dataset: SingleObjectTrackingDataset, index: int, context: _Context, iterator):
        self.dataset = dataset
        self.index = index
        self.context = context
        self.sequence = self.dataset.sequences[index]
        self.parent_iterator = iterator

    def setName(self, name: str):
        self.sequence.name = name

    def delete(self):
        self.dataset.sequences.pop(self.index)
        if self.parent_iterator is not None:
            self.parent_iterator.index -= 1
        del self

    def __len__(self):
        return len(self.sequence.frames)

    def __iter__(self):
        return SingleObjectDatasetFrameIterator(self.sequence, self.context)


class SingleObjectDatasetSequenceIterator:
    def __init__(self, dataset:SingleObjectTrackingDataset, context: _Context):
        self.dataset = dataset
        self.index = 0
        self.context = context

    def __next__(self):
        if self.index >= len(self.dataset):
            raise StopIteration
        sequence_object = SingleObjectDatasetSequenceModifier(self.dataset, self.index, self.context, self)
        self.index += 1
        return sequence_object


class SingleObjectDatasetModifier:
    def __init__(self, dataset: SingleObjectTrackingDataset):
        self.dataset = dataset
        self.context = _Context()
        self.context.root_path = pathlib.Path(self.dataset.root_path)
        self.context.has_object_category = dataset.hasAttributeCategory()
        self.context.has_fps = dataset.hasAttibuteFPS()

    def setRootPath(self, root_path: str):
        assert isinstance(root_path, str)
        self.dataset.root_path = root_path

    def setDatasetName(self, name: str):
        assert isinstance(name, str)
        self.dataset.name = name

    def applyIndicesFilter(self, indices):
        self.dataset.sequences = [self.dataset.sequences[index] for index in indices]

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        return SingleObjectDatasetSequenceIterator(self.dataset, self.context)
