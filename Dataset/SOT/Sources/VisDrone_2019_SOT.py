from Dataset.SOT.Base.constructor import SingleObjectTrackingDatasetConstructor
import os
from Dataset.DataSplit import DataSplit


def construct_VisDrone_2019_SOT(constructor: SingleObjectTrackingDatasetConstructor, seed):
    assert seed.data_split == DataSplit.Training
    sequences_path = seed.root_path
    annotations_path = seed.annotations_path

    sequences = os.listdir(sequences_path)
    sequences.sort()
    for sequence in sequences:
        constructor.beginInitializingSequence()
        constructor.setSequenceName(sequence)
        sequence_path = os.path.join(sequences_path, sequence)
        image_file_names = os.listdir(sequence_path)
        image_file_names.sort()
        for image_file_name in image_file_names:
            image_path = os.path.join(sequence_path, image_file_name)
            constructor.addFrame(image_path)
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
            constructor.setFrameAttributes(index, bounding_box)
            index += 1
        assert index == len(image_file_names)
        constructor.endInitializingSequence()
