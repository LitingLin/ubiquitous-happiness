import os
from .MOT17 import parse_2d_mot_gt_file
from Dataset.DataSplit import DataSplit


def get_mot_class_definition():
    return {
        1: 'Pedestrian',
        2: 'Person on vehicle',
        3: 'Car',
        4: 'Bicycle',
        5: 'Motorbike',
        6: 'Non motorized vehicle',
        7: 'Static person',
        8: 'Distractor',
        9: 'Occluder',
        10: 'Occluder on the ground',
        11: 'Occluder full',
        12: 'Reflection',
        13: '(Unknown)'
    }


def construct_MOT20(constructor, seed):
    assert seed.data_split == DataSplit.Full
    root_path = seed.root_path

    train_dir = os.path.join(root_path, 'train')
    sequences = os.listdir(train_dir)
    sequences.sort()

    valid_sequences = {}
    class_id_name_mapper = get_mot_class_definition()

    for sequence in sequences:
        words = sequence.split('-')
        assert len(words) == 2
        assert words[0] == 'MOT20'
        if words[1] not in valid_sequences:
            valid_sequences[words[1]] = sequence

    for sequence in valid_sequences.values():
        sequence_path = os.path.join(train_dir, sequence)

        gt_file = os.path.join(sequence_path, 'gt', 'gt.txt')
        img_path = os.path.join(sequence_path, 'img1')

        imgs = os.listdir(img_path)
        imgs.sort()
        imgs = [img for img in imgs if img.endswith('.jpg')]
        assert len(imgs) != 0

        gt_data = parse_2d_mot_gt_file(gt_file)

        constructor.beginInitializingSequence()

        words = sequence.split('-')
        constructor.setSequenceName('{}-{}'.format(words[0], words[1]))

        for img in imgs:
            constructor.addFrame(os.path.join(img_path, img))

        for object_id, (class_id, records) in gt_data.items():
            constructor.addObject(object_id, class_id_name_mapper[class_id])

            for index_of_frame, bounding_box, visibility_ratio in records:
                constructor.addRecord(index_of_frame, object_id, bounding_box, additional_attributes={'visibility_ratio': visibility_ratio})

        constructor.endInitializingSequence()
