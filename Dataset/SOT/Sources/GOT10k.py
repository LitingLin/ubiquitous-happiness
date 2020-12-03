import os
from Dataset import DataSplit
from tqdm import tqdm


def _get_category_names(root_path):
    category_names = []
    for folder in ['train', 'val']:
        for sequence_name in tqdm(open(os.path.join(root_path, folder, 'list.txt'), 'r')):
            sequence_name = sequence_name.strip()
            current_sequence_path = os.path.join(root_path, folder, sequence_name)

            objectClass = None
            for line in open(os.path.join(current_sequence_path, 'meta_info.ini'), 'r'):
                if line.startswith('object_class: '):
                    objectClass = line[len('object_class: '):]
                    objectClass = objectClass.strip()
                    if len(objectClass) == 0:
                        objectClass = None
                    break
            category_names.append(objectClass)

    category_names.sort()
    return category_names


def construct_GOT10k(constructor, seed):
    root_path = seed.root_path
    data_type = seed.data_split

    folders = []

    if data_type & DataSplit.Training:
        folders.append('train')
    if data_type & DataSplit.Validation:
        folders.append('val')

    category_names = _get_category_names(root_path)
    constructor.mergeCategoryIdNameMapper({k: v for k, v in enumerate(category_names)})
    category_name_id_map = {v: k for k, v in enumerate(category_names)}

    for folder in folders:
        for sequence_name in tqdm(open(os.path.join(root_path, folder, 'list.txt'), 'r')):
            sequence_name = sequence_name.strip()
            current_sequence_path = os.path.join(root_path, folder, sequence_name)
            images = os.listdir(current_sequence_path)
            images = [image for image in images if image.endswith('.jpg')]
            images.sort()

            is_presents = []
            for line in open(os.path.join(current_sequence_path, 'absence.label'), 'r'):
                line = line.strip()
                is_present = not bool(int(line))
                is_presents.append(is_present)

            boundingBoxes = []
            for line in open(os.path.join(current_sequence_path, 'groundtruth.txt'), 'r'):
                line = line.strip()
                boundingBox = line.split(',')
                assert len(boundingBox) == 4
                boundingBox = [float(value) for value in boundingBox]
                boundingBoxes.append(boundingBox)

            assert len(images) == len(is_presents)
            assert len(boundingBoxes) == len(images)

            objectClass = None
            for line in open(os.path.join(current_sequence_path, 'meta_info.ini'), 'r'):
                if line.startswith('object_class: '):
                    objectClass = line[len('object_class: '):]
                    objectClass = objectClass.strip()
                    if len(objectClass) == 0:
                        objectClass = None
                    break

            if objectClass is None:
                raise Exception

            constructor.beginInitializingSequence()
            constructor.setSequenceName(sequence_name)
            constructor.setSequenceObjectCategory(category_name_id_map[objectClass])

            for image, boundingBox, is_present in zip(images, boundingBoxes, is_presents):
                constructor.setFrameAttributes(
                    constructor.addFrame(os.path.join(root_path, folder, sequence_name, image)),
                    boundingBox, is_present)
            constructor.endInitializingSequence()
