import os
from Dataset.DataSplit import DataSplit


def construct_DeformSOT(constructor, seed):
    assert seed.data_split == DataSplit.Full
    root_path = seed.root_path

    auto_category_id_allocator = constructor.getAutoCategoryIdAllocationTool()

    sequences = os.listdir(root_path)
    sequences = [sequence for sequence in sequences if os.path.isdir(os.path.join(root_path, sequence)) and sequence != 'Annotations']

    groundtruth_folder = os.path.join(root_path, 'Annotations', 'gt')
    for sequence in sequences:
        sequence_dir = os.path.join(root_path, sequence)
        gt_file = os.path.join(groundtruth_folder, '{}.txt'.format(sequence))

        bounding_boxes = []

        for line in open(gt_file):
            line = line.strip()
            if len(line) == 0:
                continue

            words = line.split(',')
            assert len(words) == 4

            bounding_box = [int(v) for v in words]
            bounding_boxes.append(bounding_box)

        class_label = sequence
        for ind in reversed(range(len(class_label))):
            if class_label[ind].isdigit():
                class_label = class_label[:-1]
            else:
                break
        images = os.listdir(sequence_dir)
        images = [image for image in images if image.endswith('.jpg')]
        images.sort()

        constructor.beginInitializingSequence()
        constructor.setSequenceName(sequence)
        category_id = auto_category_id_allocator.getOrAllocateCategoryId(class_label)
        constructor.setSequenceObjectCategory(category_id)
        for image, bounding_box in zip(images, bounding_boxes):
            image_path = os.path.join(sequence_dir, image)
            constructor.setFrameAttributes(constructor.addFrame(image_path), bounding_box)
        constructor.endInitializingSequence()
