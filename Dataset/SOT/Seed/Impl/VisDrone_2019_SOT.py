import os
from Dataset.Type.data_split import DataSplit
from Dataset.SOT.Constructor.base import SingleObjectTrackingDatasetConstructor


def construct_VisDrone_2019_SOT(constructor: SingleObjectTrackingDatasetConstructor, seed):
    assert seed.data_split == DataSplit.Training
    sequences_path = seed.root_path
    annotations_path = seed.annotations_path

    sequences = os.listdir(sequences_path)
    sequences.sort()
    constructor.set_total_number_of_sequences(len(sequences))
    for sequence in sequences:
        with constructor.new_sequence() as sequence_constructor:
            sequence_constructor.set_name(sequence)
            sequence_path = os.path.join(sequences_path, sequence)
            image_file_names = os.listdir(sequence_path)
            image_file_names.sort()
            for image_file_name in image_file_names:
                image_path = os.path.join(sequence_path, image_file_name)
                with sequence_constructor.new_frame() as frame_constructor:
                    frame_constructor.set_path(image_path)
            annotation_file_name = sequence + '.txt'
            annotation_file_path = os.path.join(annotations_path, annotation_file_name)
            index = 0
            for line in open(annotation_file_path, 'r', encoding='utf-8'):
                line = line.strip()
                if len(line) == 0:
                    continue
                words = line.split(',')
                assert len(words) == 4
                bounding_box = [int(word) for word in words]
                bounding_box[0] -= 1
                bounding_box[1] -= 1
                with sequence_constructor.open_frame(index) as frame_constructor:
                    frame_constructor.set_bounding_box(bounding_box)
                index += 1
            assert index == len(image_file_names)
