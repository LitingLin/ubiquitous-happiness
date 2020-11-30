import os
from Dataset.DataSplit import DataSplit


def construct_DTB70(constructor, seed):
    assert seed.data_split == DataSplit.Full
    root_path = seed.root_path

    sequence_list = os.listdir(root_path)
    sequence_list = [dirname for dirname in sequence_list if os.path.isdir(os.path.join(root_path, dirname))]
    sequence_list.sort()

    number_of_sequences = len(sequence_list)

    for index in range(number_of_sequences):
        sequence_name = sequence_list[index]

        path = os.path.join(root_path, sequence_name)
        img_path = os.path.join(path, 'img')

        ground_truth_files = [file for file in os.listdir(path) if file.endswith('groundtruth_rect.txt')]
        assert len(ground_truth_files) == 1

        images = os.listdir(img_path)
        images = [image for image in images if image.endswith('.jpg')]
        images.sort()

        constructor.beginInitializingSequence()
        constructor.setSequenceName(sequence_name)

        class_label = sequence_name
        for ind in reversed(range(len(class_label))):
            if class_label[ind].isdigit():
                class_label = class_label[:-1]
            else:
                break
        constructor.setSequenceObjectCategory(class_label)

        for line_index, line in enumerate(open(os.path.join(path, ground_truth_files[0]), 'r')):
            line = line.strip()
            bounding_box = [float(value) for value in line.split(',') if value]
            constructor.setFrameAttributes(constructor.addFrame(os.path.join(img_path, images[line_index])), bounding_box)

        constructor.endInitializingSequence()
