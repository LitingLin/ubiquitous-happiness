import os
from Dataset.Type.data_split import DataSplit
from Dataset.SOT.Constructor.base import SingleObjectTrackingDatasetConstructor


def _get_class_name(sequence: str):
    class_name = []
    for c in sequence:
        if c.isdigit():
            continue
        c = c.lower()
        class_name.append(c)
    return ''.join(class_name)


_category_id_name_map = {0: 'animal', 1: 'basketball', 2: 'bmx', 3: 'car', 4: 'chasingdrones', 5: 'girl', 6: 'gull', 7: 'horse', 8: 'kiting', 9: 'manrunning', 10: 'motor', 11: 'mountainbike', 12: 'paragliding', 13: 'racecar', 14: 'rccar', 15: 'sheep', 16: 'skateboarding', 17: 'skiing', 18: 'snowboarding', 19: 'soccer', 20: 'speedcar', 21: 'streetbasketball', 22: 'sup', 23: 'surfing', 24: 'vaulting', 25: 'wakeboarding', 26: 'walking', 27: 'yacht', 28: 'zebra'}


def construct_DTB70(constructor: SingleObjectTrackingDatasetConstructor, seed):
    assert seed.data_split == DataSplit.Full
    root_path = seed.root_path

    sequence_list = os.listdir(root_path)
    sequence_list = [dirname for dirname in sequence_list if os.path.isdir(os.path.join(root_path, dirname))]
    sequence_list.sort()
    number_of_sequences = len(sequence_list)
    constructor.set_total_number_of_sequences(number_of_sequences)
    constructor.set_category_id_name_map(_category_id_name_map)
    category_name_id_map = {v:k for k, v in _category_id_name_map.items()}

    for index in range(number_of_sequences):
        sequence_name = sequence_list[index]

        path = os.path.join(root_path, sequence_name)
        img_path = os.path.join(path, 'img')

        ground_truth_files = [file for file in os.listdir(path) if file.endswith('groundtruth_rect.txt')]
        assert len(ground_truth_files) == 1

        images = os.listdir(img_path)
        images = [image for image in images if image.endswith('.jpg')]
        images.sort()

        with constructor.new_sequence(category_name_id_map[_get_class_name(sequence_name)]) as sequence_constructor:
            sequence_constructor.set_name(sequence_name)

            for line_index, line in enumerate(open(os.path.join(path, ground_truth_files[0]), 'r')):
                line = line.strip()
                bounding_box = [float(value) for value in line.split(',') if value]
                with sequence_constructor.new_frame() as frame_constructor:
                    frame_constructor.set_path(os.path.join(img_path, images[line_index]))
                    frame_constructor.set_bounding_box(bounding_box)
