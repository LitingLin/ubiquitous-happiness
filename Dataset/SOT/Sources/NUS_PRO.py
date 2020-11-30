import os
from Dataset.DataSplit import DataSplit


def construct_NUSPRO(constructor, seed):
    assert seed.data_split == DataSplit.Full
    root_path = seed.root_path

    sequences = os.listdir(root_path)
    sequences = [sequence for sequence in sequences if os.path.isdir(os.path.join(root_path, sequence))]
    sequences.sort()
    for sequence in sequences:
        sequence_path = os.path.join(root_path, sequence)
        class_name = sequence[:-4]

        files = os.listdir(sequence_path)
        images = [image for image in files if image.endswith('.jpg')]
        images.sort()

        occlusion_file = os.path.join(sequence_path, 'occlusion.txt')
        # patch:
        if sequence == 'helicopter_011':
            occlusions = [False] * 300
        else:
            occlusions = []
            for line in open(occlusion_file):
                line = line.strip()
                if len(line) == 0:
                    continue
                occlusions.append(bool(int(line)))

        groundtruth_file = os.path.join(sequence_path, 'groundtruth.txt')
        bounding_boxes = []
        for line in open(groundtruth_file):
            line = line.strip()
            if len(line) == 0:
                continue
            words = line.split(' ')
            assert len(words) == 4
            bounding_box = [int(words[0]), int(words[1]), int(words[2]) - int(words[0]), int(words[3]) - int(words[1])]
            bounding_boxes.append(bounding_box)

        constructor.beginInitializingSequence()
        constructor.setSequenceName(sequence)
        if class_name == 'basketball':
            class_name = 'person'
        elif class_name == 'gymnastics':
            class_name = 'person'
        elif class_name == 'handball':
            class_name = 'person'
        elif class_name == 'hat':
            class_name = 'face'
        elif class_name == 'interview':
            class_name = 'face'
        elif class_name == 'mask':
            class_name = 'face'
        elif class_name == 'motorcycle':
            class_name = 'motorcycle person'
        elif class_name == 'pedestrian':
            class_name = 'person'
        elif class_name == 'politician':
            class_name = 'face'
        elif class_name == 'racing':
            class_name = 'person'
        elif class_name == 'soccer':
            class_name = 'person'
        elif class_name == 'sunglasses':
            class_name = 'face'
        elif class_name == 'tennis':
            class_name = 'person'
        if sequence == 'long_seq_001':
            class_name = 'person'
        elif sequence == 'long_seq_002':
            class_name = 'car'
        elif sequence == 'long_seq_003':
            class_name = 'airplane'
        elif sequence == 'long_seq_004':
            class_name = 'person'
        elif sequence == 'long_seq_005':
            class_name = 'person'

        constructor.setSequenceObjectCategory(class_name)

        assert len(occlusions) == len(bounding_boxes)
        assert len(bounding_boxes) == len(images)
        for image, bounding_box, occlusion in zip(images, bounding_boxes, occlusions):
            image_path = os.path.join(sequence_path, image)
            constructor.setFrameAttributes(constructor.addFrame(image_path), bounding_box, ~occlusion)
        constructor.endInitializingSequence()
