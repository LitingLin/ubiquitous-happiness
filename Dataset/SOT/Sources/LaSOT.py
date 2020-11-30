from Dataset import DataSplit
import os


def construct_LaSOT(constructor, seed):
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

    for class_name in class_names:
        if not any(sequenceName.startswith(class_name) for sequenceName in validSequenceNames):
            raise Exception

        class_path = os.path.join(root_path, class_name)
        sequence_names = os.listdir(class_path)
        sequence_names = [sequence_name for sequence_name in sequence_names if
                          os.path.isdir(os.path.join(class_path, sequence_name))]
        sequence_names.sort()

        for sequence_name in sequence_names:
            if sequence_name not in validSequenceNames:
                continue
            constructor.beginInitializingSequence()
            constructor.setSequenceName(sequence_name)
            constructor.setSequenceObjectCategory(class_name)
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
                constructor.setFrameAttributes(constructor.addFrame(image_path), bounding_box, not(is_fully_occlusion | is_out_of_view))
                constructor.setSequenceAttribute('occlusion', is_fully_occlusion)
                constructor.setSequenceAttribute('out of view', is_out_of_view)
            constructor.endInitializingSequence()
