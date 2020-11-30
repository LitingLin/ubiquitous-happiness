import os
from ..FactorySeeds.NFS import NFSDatasetVersionFlag
from Dataset.DataSplit import DataSplit


def construct_NFS(constructor, seed):
    assert seed.data_split == DataSplit.Full
    root_path = seed.root_path
    version = seed.version

    sequence_list = os.listdir(root_path)
    sequence_list = [dirname for dirname in sequence_list if os.path.isdir(os.path.join(root_path, dirname))]
    sequence_list = [dirname for dirname in sequence_list if
                 os.path.exists(os.path.join(root_path, dirname, '30', dirname)) and os.path.exists(
                     os.path.join(root_path, dirname, '240', dirname))]

    if version == NFSDatasetVersionFlag.fps_30:
        subDirName = '30'
    elif version == NFSDatasetVersionFlag.fps_240:
        subDirName = '240'
    else:
        raise Exception

    records = {}

    for sequence_name in sequence_list:
        sequence_images_path = os.path.join(root_path, sequence_name, subDirName, sequence_name)
        sequence_anno_file_path = os.path.join(root_path, sequence_name, subDirName, '{}.txt'.format(sequence_name))
        images = os.listdir(sequence_images_path)
        images = [image for image in images if image.endswith('.jpg')]
        images.sort()

        bounding_boxes = []
        className = None

        for line_count, line in enumerate(open(sequence_anno_file_path)):
            if subDirName == '30':
                if line_count % 8 != 0:
                    continue
            line = line.strip()
            first_quote_index = line.find('"')
            if first_quote_index == -1:
                raise Exception
            attributes = line[:first_quote_index].split()
            bbox = attributes[1:5]
            bbox = [int(v) for v in bbox]
            bbox[2] = bbox[2] - bbox[0]
            bbox[3] = bbox[3] - bbox[1]
            bounding_boxes.append(bbox)
            second_quote_index = line.rfind('"')
            if second_quote_index == -1 or second_quote_index <= first_quote_index:
                raise Exception
            current_class_name = line[first_quote_index + 1: second_quote_index]
            if className is None:
                className = current_class_name
            elif className != current_class_name:
                raise Exception

        len_diff = abs(len(images) - len(bounding_boxes))
        if len_diff != 0:
            length = min(len(images), len(bounding_boxes))
            images = images[:length]
            bounding_boxes = bounding_boxes[:length]

        if className not in records:
            records[className] = []
        records[className].append((sequence_name, images, bounding_boxes))

    for class_name, sequences in records.items():
        for sequence_name, images, bounding_boxes in sequences:
            current_sequence_image_path = os.path.join(sequence_name, subDirName, sequence_name)
            constructor.beginInitializingSequence()
            constructor.setSequenceName(sequence_name)
            constructor.setSequenceObjectCategory(class_name)
            for image, bounding_box in zip(images, bounding_boxes):
                constructor.setFrameAttributes(constructor.addFrame(os.path.join(root_path, current_sequence_image_path, image)), bounding_box)
            constructor.endInitializingSequence()
