from Dataset.SOT.Constructor.base import SingleObjectTrackingDatasetConstructor
import os
from Dataset.Type.data_split import DataSplit

_category_id_name_map = {0: 'atv', 1: 'badminton', 2: 'cosplay', 3: 'dancingshoe', 4: 'footbag', 5: 'frisbee', 6: 'jianzi', 7: 'lantern', 8: 'misc', 9: 'opossum', 10: 'paddle', 11: 'raccoon', 12: 'rhino', 13: 'skatingshoe', 14: 'wingsuit'}


def construct_LaSOT_Extension(constructor: SingleObjectTrackingDatasetConstructor, seed):
    assert seed.data_split == DataSplit.Full
    root_path = seed.root_path

    class_names = os.listdir(root_path)
    class_names = [class_name for class_name in class_names if os.path.isdir(os.path.join(root_path, class_name))]
    class_names.sort()
    constructor.set_category_id_name_map(_category_id_name_map)
    category_name_id_map = {v: k for k, v in _category_id_name_map.items()}

    sequences = []
    for class_name in class_names:
        class_path = os.path.join(root_path, class_name)
        sequence_names = os.listdir(class_path)
        sequence_names = [sequence_name for sequence_name in sequence_names if
                          os.path.isdir(os.path.join(class_path, sequence_name))]
        sequence_names.sort()

        for sequence_name in sequence_names:
            sequence_path = os.path.join(class_path, sequence_name)
            groundtruth_file_path = os.path.join(sequence_path, 'groundtruth.txt')
            sequences.append((class_name, sequence_name, sequence_path, groundtruth_file_path))

    constructor.set_total_number_of_sequences(len(sequences))

    for class_name, sequence_name, sequence_path, groundtruth_file_path in sequences:
        category_id = category_name_id_map[class_name]
        with constructor.new_sequence(category_id) as sequence_constructor:
            sequence_constructor.set_name(sequence_name)
            bounding_boxes = []
            with open(groundtruth_file_path, 'rb') as fid:
                for line in fid:
                    line = line.decode('UTF-8')
                    line = line.strip()
                    words = line.split(',')
                    if len(words) != 4:
                        raise Exception('error in parsing file {}'.format(groundtruth_file_path))
                    bounding_box = [int(words[0]), int(words[1]), int(words[2]), int(words[3])]
                    bounding_boxes.append(bounding_box)
            full_occlusion_file_path = os.path.join(sequence_path, 'full_occlusion.txt')
            with open(full_occlusion_file_path, 'rb') as fid:
                file_content = fid.read().decode('UTF-8')
                file_content = file_content.strip()
                words = file_content.split(',')
                is_fully_occlusions = [word == '1' for word in words]
            out_of_view_file_path = os.path.join(sequence_path, 'out_of_view.txt')
            with open(out_of_view_file_path, 'rb') as fid:
                file_content = fid.read().decode('UTF-8')
                file_content = file_content.strip()
                words = file_content.split(',')
                is_out_of_views = [word == '1' for word in words]
            images_path = os.path.join(sequence_path, 'img')
            if len(bounding_boxes) != len(is_fully_occlusions) or len(is_fully_occlusions) != len(is_out_of_views):
                raise Exception('annotation length mismatch in {}'.format(sequence_path))
            for index in range(len(bounding_boxes)):
                image_file_name = '{:0>8d}.jpg'.format(index + 1)
                image_path = os.path.join(images_path, image_file_name)
                if not os.path.exists(image_path):
                    raise Exception('file not exists: {}'.format(image_path))
                bounding_box = bounding_boxes[index]
                is_fully_occlusion = is_fully_occlusions[index]
                is_out_of_view = is_out_of_views[index]
                with sequence_constructor.new_frame() as frame_constructor:
                    frame_constructor.set_path(image_path)
                    frame_constructor.set_bounding_box(bounding_box, validity=not(is_fully_occlusion | is_out_of_view))
                    frame_constructor.set_object_attribute('occlusion', is_fully_occlusion)
                    frame_constructor.set_object_attribute('out of view', is_out_of_view)
