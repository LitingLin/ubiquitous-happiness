import os
import json
from Dataset.DataSplit import DataSplit


def construct_PTB(constructor, seed):
    assert seed.data_split == DataSplit.Full
    root_path = seed.root_path

    sequences = os.listdir(root_path)
    sequences = [sequence for sequence in sequences if os.path.isdir(os.path.join(root_path, sequence))]
    sequences.sort()

    for sequence in sequences:
        current_dir = os.path.join(root_path, sequence)

        bbox_groundtruth_file = os.path.join(current_dir, '{}.txt'.format(sequence))
        rgb_image_file_dir = os.path.join(current_dir, 'rgb')
        frame_description_file = os.path.join(current_dir, 'frames.json')

        with open(frame_description_file, 'rb') as fid:
            frame_information = json.load(fid)

        image_timestamps = frame_information['imageTimestamp']
        image_frame_ids = frame_information['imageFrameID']

        constructor.beginInitializingSequence()
        constructor.setSequenceName(sequence)
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

        constructor.setSequenceObjectCategory(category)

        count = 0
        for line in open(bbox_groundtruth_file, 'r'):
            words = line.split(',')
            words = [word.strip() for word in words]

            image_path = os.path.join(rgb_image_file_dir, 'r-{}-{}.png'.format(image_timestamps[count], image_frame_ids[count]))
            if words[0] == 'NaN':
                constructor.setFrameAttributes(constructor.addFrame(image_path), None, False)
            else:
                bounding_box = [int(words[0]), int(words[1]), int(words[2]), int(words[3])]
                constructor.setFrameAttributes(constructor.addFrame(image_path), bounding_box, True)
            count += 1

        constructor.endInitializingSequence()
