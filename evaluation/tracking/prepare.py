from Dataset.SOT.Storage.MemoryMapped.dataset import SingleObjectTrackingDataset_MemoryMapped
from typing import List
from Miscellaneous.most_frequent import get_most_frequent_item_from_list


def prepare_sequences(datasets: List[SingleObjectTrackingDataset_MemoryMapped], remove_duplicate_sequences_by_name=True, sort_by_sequence_frame_size=True):
    if remove_duplicate_sequences_by_name:
        sequences = {}
        for dataset in datasets:
            for sequence in dataset:
                if sequence.get_name() not in sequences:
                    sequences[sequence.get_name()] = sequence
        sequences = list(sequences.values())
    else:
        sequences = [sequence for dataset in datasets for sequence in dataset]

    if sort_by_sequence_frame_size:
        sequence_with_frame_size = []
        for sequence in sequences:
            frame_sizes = []
            for frame in sequence:
                image_size = frame.get_image_size()
                image_size = image_size[0] * image_size[1]
                frame_sizes.append(image_size)
            frame_size = get_most_frequent_item_from_list(frame_sizes)
            sequence_with_frame_size.append((sequence, frame_size))
        sequence_with_frame_size = sorted(sequence_with_frame_size, key=lambda x: x[1])
        sequences = [s[0] for s in sequence_with_frame_size]
    return sequences
