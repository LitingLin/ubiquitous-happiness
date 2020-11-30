from Dataset.MOT.Base.dataset import MultipleObjectTrackingDataset
from ._impl import _set_image_path, _get_or_allocate_category_id


class MultipleObjectTrackingDatasetInFrameObjectModifier:
    def __init__(self, frame_index, frame, object_id, object_attributes, object_, iterator=None):
        self.frame_index = frame_index
        self.frame = frame
        self.object_id = object_id
        self.object_attributes = object_attributes
        self.object_ = object_
        self.parent_iterator = iterator

    def getFrameIndex(self):
        return self.frame_index

    def getImageSize(self):
        return self.frame.size

    def delete(self):
        self.frame.objects.pop(self.object_id)
        self.object_attributes.frame_indices.remove(self.frame_index)
        if self.parent_iterator is not None:
            self.parent_iterator.pop_current()
        del self

    def getObjectId(self):
        return self.object_id

    def setBoundingBox(self, bounding_box):
        self.object_.bounding_box = bounding_box

    def getBoundingBox(self):
        return self.object_.bounding_box

    def setAttributeIsPresent(self, value: bool):
        self.object_.is_present = value


class MultipleObjectTrackingDatasetInFrameObjectIterator:
    def __init__(self, frame_index, frame, object_id_attributes_mapper):
        self.frame_index = frame_index
        self.index = 0
        self.frame = frame
        self.object_ids = list(self.frame.objects.keys())
        self.object_id_attributes_mapper = object_id_attributes_mapper

    def pop_current(self):
        self.index -= 1
        self.object_ids.pop(self.index)

    def __next__(self):
        if self.index >= len(self.object_ids):
            raise StopIteration

        object_id = self.object_ids[self.index]
        in_frame_object_modifier = MultipleObjectTrackingDatasetInFrameObjectModifier(self.frame_index, self.frame, object_id, self.object_id_attributes_mapper[object_id], self.frame.objects[object_id], self)
        self.index += 1
        return in_frame_object_modifier


class MultipleObjectTrackingDatasetFrameModifier:
    def __init__(self, dataset, sequence, index, iterator=None):
        self.dataset = dataset
        self.sequence = sequence
        self.index = index
        self.frame = sequence.frames[index]
        self.iterator = iterator

    def setImagePath(self, path: str):
        self.frame.size, self.frame.image_path = _set_image_path(self.dataset.root_path, path)

    def getImageSize(self):
        return self.frame.size

    def __iter__(self):
        return MultipleObjectTrackingDatasetInFrameObjectIterator(self.index, self.frame, self.sequence.object_id_attributes_mapper)

    def __len__(self):
        return len(self.frame.objects)

    def delete(self):
        for track_info in self.sequence.object_id_attributes_mapper.values():
            frame_indices: list = track_info.frame_indices
            to_be_popped = None
            for index, frame_index in enumerate(frame_indices):
                if frame_index == self.index:
                    to_be_popped = index
                if frame_index > self.index:
                    frame_indices[index] -= 1
            if to_be_popped is not None:
                frame_indices.pop(to_be_popped)
        self.sequence.frames.pop(self.index)

        if self.iterator is not None:
            self.iterator.index -= 1
        del self


class MultipleObjectTrackingDatasetFrameIterator:
    def __init__(self, dataset, sequence):
        self.dataset = dataset
        self.sequence = sequence
        self.index = 0

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index >= len(self.sequence.frames):
            raise StopIteration

        modifier = MultipleObjectTrackingDatasetFrameModifier(self.dataset, self.sequence, self.index, self)
        self.index += 1
        return modifier


class MultipleObjectTrackingDatasetObjectTrackFrameIterator:
    def __init__(self, sequence, object_id):
        self.sequence = sequence
        self.object_id = object_id
        self.object_attributes = sequence.object_id_attributes_mapper[object_id]
        self.index = 0

    def pop_current(self):
        self.index -= 1

    def __next__(self):
        if self.index >= len(self.object_attributes.frame_indices):
            raise StopIteration

        frame_index = self.object_attributes.frame_indices[self.index]
        frame = self.sequence.frames[frame_index]
        in_frame_object_modifier = MultipleObjectTrackingDatasetInFrameObjectModifier(frame_index, frame, self.object_id, self.object_attributes, frame.objects[self.object_id], self)
        self.index += 1
        return in_frame_object_modifier


class MultipleObjectTrackingDatasetObjectTrackModifier:
    def __init__(self, dataset, sequence, object_id: int, iterator=None):
        self.dataset = dataset
        self.sequence = sequence
        self.object_id = object_id
        self.object_attributes = sequence.object_id_attributes_mapper[object_id]
        self.parent_iterator = iterator

    def __iter__(self):
        return MultipleObjectTrackingDatasetObjectTrackFrameIterator(self.sequence, self.object_id)

    def setCategoryName(self, name: str):
        self.object_attributes.category_id = _get_or_allocate_category_id(name, self.dataset.category_names, self.dataset.category_name_id_mapper)

    def getObjectId(self):
        return self.object_id

    def delete(self):
        for frame in self.sequence.frames:
            if self.object_id in frame.objects:
                frame.objects.pop(self.object_id)

        self.sequence.object_id_attributes_mapper.pop(self.object_id)
        self.sequence.object_ids.remove(self.object_id)

        if self.parent_iterator is not None:
            self.parent_iterator.index -= 1
        del self

    def __len__(self):
        return len(self.object_attributes.frame_indices)


class MultipleObjectTrackingDatasetObjectTrackIterator:
    def __init__(self, dataset, sequence):
        self.dataset = dataset
        self.sequence = sequence
        self.index = 0

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index >= len(self.sequence.object_ids):
            raise StopIteration

        object_id = self.sequence.object_ids[self.index]
        object_track_modifier = MultipleObjectTrackingDatasetObjectTrackModifier(self.dataset, self.sequence, object_id, self)
        self.index += 1
        return object_track_modifier


class MultipleObjectTrackingDatasetSequenceModifier:
    def __init__(self, dataset: MultipleObjectTrackingDataset, index: int, iterator=None):
        self.dataset = dataset
        self.index = index
        self.sequence = dataset.sequences[index]
        self.parent_iterator = iterator

    def setName(self, name):
        self.sequence.name = name

    def iterateByObjectTrack(self):
        return MultipleObjectTrackingDatasetObjectTrackIterator(self.dataset, self.sequence)

    def iterateByFrame(self):
        return MultipleObjectTrackingDatasetFrameIterator(self.dataset, self.sequence)

    def __len__(self):
        return len(self.sequence.frames)

    def delete(self):
        self.dataset.sequences.pop(self.index)
        if self.parent_iterator is not None:
            self.parent_iterator.index -= 1
        del self


class MultipleObjectTrackingDatasetSequenceIterator:
    def __init__(self, dataset):
        self.dataset = dataset
        self.index = 0

    def __next__(self):
        if self.index >= len(self.dataset.sequences):
            raise StopIteration
        modifier = MultipleObjectTrackingDatasetSequenceModifier(self.dataset, self.index, self)
        self.index += 1
        return modifier


class MultipleObjectTrackingDatasetModifier:
    def __init__(self, dataset: MultipleObjectTrackingDataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        return MultipleObjectTrackingDatasetSequenceIterator(self.dataset)

    def setRootPath(self, path: str):
        self.dataset.root_path = path

    def setName(self, name: str):
        self.dataset.name = name
