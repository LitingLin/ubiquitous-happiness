from Dataset.MOT.Storage.MemoryMapped.dataset import MultipleObjectTrackingDatasetSequence_MemoryMapped, MultipleObjectTrackingDatasetSequenceObject_MemoryMapped


class _MOTTrackIterator:
    def __init__(self, data_getter, sequence: MultipleObjectTrackingDatasetSequence_MemoryMapped, sequence_object: MultipleObjectTrackingDatasetSequenceObject_MemoryMapped):
        self.data_getter = data_getter
        self.sequence = sequence
        self.sequence_object = sequence_object
        self.frame_indices = sequence_object.get_all_frame_index()
        self.frame_index_iter = iter(range(min(self.frame_indices), max(self.frame_indices) + 1))

    def __next__(self):
        frame_index = next(self.frame_index_iter)
        frame = self.sequence[frame_index]
        if frame_index in self.frame_indices:
            frame_object = frame.get_object_by_id(self.sequence_object.get_id())
        else:
            frame_object = None
        return self.data_getter(self.sequence, frame, self.sequence_object, frame_object)


class MultipleObjectTrackingDatasetTrackIteratorGenerator:
    def __init__(self, data_getter, sequence: MultipleObjectTrackingDatasetSequence_MemoryMapped, sequence_object: MultipleObjectTrackingDatasetSequenceObject_MemoryMapped):
        self.data_getter = data_getter
        self.sequence = sequence
        self.sequence_object = sequence_object

    def __iter__(self):
        return _MOTTrackIterator(self.data_getter, self.sequence, self.sequence_object)

    def __len__(self):
        frame_indices = self.sequence_object.get_all_frame_index()
        return max(frame_indices) - min(frame_indices) + 1
