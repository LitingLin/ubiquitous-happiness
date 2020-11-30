import os
from Dataset.DataSplit import DataSplit


def construct_TrackingNet(constructor, seed):
    root_path = seed.root_path
    data_type = seed.data_split
    enable_set_ids = seed.enable_set_ids
    sequence_name_class_map_file_path = seed.sequence_name_class_map_file_path
    if data_type != DataSplit.Training and enable_set_ids is not None:
        raise Exception("unsupported configuration")

    sequence_name_class_map = {}

    if sequence_name_class_map_file_path is None:
        sequence_name_class_map_file_path = os.path.join(root_path, 'sequence_classes_map.txt')

    for line in open(sequence_name_class_map_file_path, 'r', encoding='utf-8'):
        line = line.strip()
        name, category = line.split('\t')
        sequence_name_class_map[name] = category

    if enable_set_ids is not None:
        trackingNetSubsets = ['TRAIN_{}'.format(v) for v in enable_set_ids]
    else:
        trackingNetSubsets = []
        if data_type & DataSplit.Training:
            trackingNetSubsets = ['TRAIN_{}'.format(v) for v in range(12)]
        if data_type & DataSplit.Testing:
            trackingNetSubsets.append('TEST')

    for subset in trackingNetSubsets:
        subset_path = os.path.join(root_path, subset)
        frames_path = os.path.join(subset_path, 'frames')
        anno_path = os.path.join(subset_path, 'anno')

        bounding_box_annotation_files = os.listdir(anno_path)
        bounding_box_annotation_files = [bounding_box_annotation_file for bounding_box_annotation_file in
                                         bounding_box_annotation_files if bounding_box_annotation_file.endswith('.txt')]
        bounding_box_annotation_files.sort()

        sequences = [sequence[:-4] for sequence in bounding_box_annotation_files]
        for sequence, bounding_box_annotation_file in zip(sequences, bounding_box_annotation_files):
            constructor.beginInitializingSequence()
            constructor.setSequenceName(sequence)
            constructor.setSequenceObjectCategory(sequence_name_class_map[sequence])

            sequence_image_path = os.path.join(frames_path, sequence)
            bounding_box_annotation_file_path = os.path.join(anno_path, bounding_box_annotation_file)

            bounding_boxes = []
            for line in open(bounding_box_annotation_file_path, 'rb'):
                line = line.strip()
                if len(line) > 0:
                    values = line.split(b',')
                    assert len(values) == 4
                    bounding_boxes.append([float(values[0]), float(values[1]), float(values[2]), float(values[3])])
            images = os.listdir(sequence_image_path)
            images = [image for image in images if image.endswith('.jpg')]
            assert len(images) == len(bounding_boxes)

            for i in range(len(images)):
                image_file_name = '{}.jpg'.format(i)
                image_file_path = os.path.join(sequence_image_path, image_file_name)
                constructor.addFrame(image_file_path)
                constructor.setFrameAttributes(i, bounding_boxes[i])
            constructor.endInitializingSequence()
