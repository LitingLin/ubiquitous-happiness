from Dataset.Type.data_split import DataSplit
import os
from Dataset.SOT.Constructor.base import SingleObjectTrackingDatasetConstructor
from Miscellaneous.natural_keys import natural_keys


_category_id_name_map = {0: 'airplane', 1: 'basketball', 2: 'bear', 3: 'bicycle', 4: 'bird', 5: 'boat', 6: 'book', 7: 'bottle', 8: 'bus', 9: 'car', 10: 'cat', 11: 'cattle', 12: 'chameleon', 13: 'coin', 14: 'crab', 15: 'crocodile', 16: 'cup', 17: 'deer', 18: 'dog', 19: 'drone', 20: 'electricfan', 21: 'elephant', 22: 'flag', 23: 'fox', 24: 'frog', 25: 'gametarget', 26: 'gecko', 27: 'giraffe', 28: 'goldfish', 29: 'gorilla', 30: 'guitar', 31: 'hand', 32: 'hat', 33: 'helmet', 34: 'hippo', 35: 'horse', 36: 'kangaroo', 37: 'kite', 38: 'leopard', 39: 'licenseplate', 40: 'lion', 41: 'lizard', 42: 'microphone', 43: 'monkey', 44: 'motorcycle', 45: 'mouse', 46: 'person', 47: 'pig', 48: 'pool', 49: 'rabbit', 50: 'racing', 51: 'robot', 52: 'rubicCube', 53: 'sepia', 54: 'shark', 55: 'sheep', 56: 'skateboard', 57: 'spider', 58: 'squirrel', 59: 'surfboard', 60: 'swing', 61: 'tank', 62: 'tiger', 63: 'train', 64: 'truck', 65: 'turtle', 66: 'umbrella', 67: 'volleyball', 68: 'yoyo', 69: 'zebra'}


def construct_LaSOT(constructor: SingleObjectTrackingDatasetConstructor, seed):
    root_path = seed.root_path
    data_type = seed.data_split

    validSequenceNames = []

    if data_type & DataSplit.Training:
        with open(os.path.join(root_path, 'training_set.txt'), 'rb') as fid:
            for textLine in fid:
                textLine = textLine.decode('UTF-8')
                textLine = textLine.strip()
                validSequenceNames.append(textLine)

    if data_type & DataSplit.Validation:
        with open(os.path.join(root_path, 'testing_set.txt'), 'rb') as fid:
            for textLine in fid:
                textLine = textLine.decode('UTF-8')
                textLine = textLine.strip()
                validSequenceNames.append(textLine)

    class_names = os.listdir(root_path)
    class_names = [class_name for class_name in class_names if os.path.isdir(os.path.join(root_path, class_name))]
    class_names.sort()

    constructor.set_category_id_name_map(_category_id_name_map)
    category_name_id_map = {v: k for k, v in _category_id_name_map.items()}

    tasks = []

    for class_name in class_names:
        if not any(sequenceName.startswith(class_name) for sequenceName in validSequenceNames):
            raise Exception

        class_path = os.path.join(root_path, class_name)
        sequence_names = os.listdir(class_path)
        sequence_names = [sequence_name for sequence_name in sequence_names if
                          os.path.isdir(os.path.join(class_path, sequence_name))]
        sequence_names.sort(key=natural_keys)

        for sequence_name in sequence_names:
            if sequence_name not in validSequenceNames:
                continue
            tasks.append((class_name, sequence_name))

    constructor.set_total_number_of_sequences(len(tasks))

    for class_name, sequence_name in tasks:
        category_id = category_name_id_map[class_name]
        class_path = os.path.join(root_path, class_name)

        with constructor.new_sequence(category_id) as sequence_constructor:
            sequence_constructor.set_name(sequence_name)

            sequence_path = os.path.join(class_path, sequence_name)
            groundtruth_file_path = os.path.join(sequence_path, 'groundtruth.txt')
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
                    frame_constructor.set_bounding_box(bounding_box, validity=not (is_fully_occlusion | is_out_of_view))
                    frame_constructor.set_object_attribute('occlusion', is_fully_occlusion)
                    frame_constructor.set_object_attribute('out of view', is_out_of_view)
