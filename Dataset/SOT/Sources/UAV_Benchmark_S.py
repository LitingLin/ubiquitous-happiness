import os
from Dataset.DataSplit import DataSplit


def construct_UAVBenchmarkS(constructor, seed):
    assert seed.data_split == DataSplit.Full
    root_path = seed.root_path
    annotation_path = seed.annotation_path

    sequences = os.listdir(root_path)
    sequences = [sequence for sequence in sequences if os.path.isdir(os.path.join(root_path, sequence)) and sequence != 'UAV-benchmark-SOT_v1.0']
    sequences.sort()
    if annotation_path is None:
        groundtruth_path = os.path.join(root_path, 'UAV-benchmark-SOT_v1.0', 'anno')
    else:
        groundtruth_path = annotation_path

    for sequence in sequences:
        sequence_path = os.path.join(root_path, sequence)
        images = os.listdir(sequence_path)
        images.sort()
        gt_file = '{}_gt.txt'.format(sequence)
        gt_file = os.path.join(groundtruth_path, gt_file)
        bounding_boxes = []
        for line in open(gt_file):
            line = line.strip()
            if len(line) == 0:
                continue
            words = line.split(',')
            assert len(words) == 4
            bounding_box = [int(words[0]) - 1, int(words[1]) - 1, int(words[2]), int(words[3])]
            bounding_boxes.append(bounding_box)

        assert len(images) == len(bounding_boxes)

        constructor.beginInitializingSequence()
        constructor.setSequenceName(sequence)
        constructor.setSequenceObjectCategory('(Unknown)')

        for image, bounding_box in zip(images, bounding_boxes):
            image_path = os.path.join(sequence_path, image)
            constructor.setFrameAttributes(constructor.addFrame(image_path), bounding_box)

        constructor.endInitializingSequence()
