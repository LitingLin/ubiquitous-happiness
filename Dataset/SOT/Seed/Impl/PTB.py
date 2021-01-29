import os
import json
from Dataset.Type.data_split import DataSplit
from Dataset.SOT.Constructor.base import SingleObjectTrackingDatasetConstructor


_category_id_name_map = {0: 'toy', 1: 'child', 2: 'face', 3: 'person', 4: 'cup'}


def construct_PTB(constructor: SingleObjectTrackingDatasetConstructor, seed):
    assert seed.data_split == DataSplit.Full
    root_path = seed.root_path

    sequences = os.listdir(root_path)
    sequences = [sequence for sequence in sequences if os.path.isdir(os.path.join(root_path, sequence))]
    sequences.sort()

    category_name_id_map = {v: k for k, v in _category_id_name_map.items()}
    constructor.set_total_number_of_sequences(len(sequences))
    constructor.set_category_id_name_map(_category_id_name_map)
    for sequence in sequences:
        current_dir = os.path.join(root_path, sequence)

        bbox_groundtruth_file = os.path.join(current_dir, '{}.txt'.format(sequence))
        rgb_image_file_dir = os.path.join(current_dir, 'rgb')
        frame_description_file = os.path.join(current_dir, 'frames.json')

        with open(frame_description_file, 'rb') as fid:
            frame_information = json.load(fid)

        image_timestamps = frame_information['imageTimestamp']
        image_frame_ids = frame_information['imageFrameID']

        if sequence == 'bear_front':
            category = 'toy'
        elif sequence == 'child_no1':
            category = 'child'
        elif sequence == 'face_occ5':
            category = 'face'
        elif sequence == 'new_ex_occ4':
            category = 'person'
        elif sequence == 'zcup_move_1':
            category = 'cup'
        else:
            raise Exception

        with constructor.new_sequence(category_name_id_map[category]) as sequence_constructor:
            sequence_constructor.set_name(sequence)

            count = 0
            for line in open(bbox_groundtruth_file, 'r'):
                words = line.split(',')
                words = [word.strip() for word in words]

                image_path = os.path.join(rgb_image_file_dir, 'r-{}-{}.png'.format(image_timestamps[count], image_frame_ids[count]))
                with sequence_constructor.new_frame() as frame_constructor:
                    frame_constructor.set_path(image_path)
                if words[0] != 'NaN':
                    bounding_box = [int(words[0]), int(words[1]), int(words[2]), int(words[3])]
                    frame_constructor.set_bounding_box(bounding_box)
                count += 1
