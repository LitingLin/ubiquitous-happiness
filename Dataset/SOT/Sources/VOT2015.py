import os


def constructVOT2015(constructor, seed):
    root_path = seed.root_path
    sequence_list = os.listdir(root_path)
    sequence_list = [dirname for dirname in sequence_list if os.path.isdir(os.path.join(root_path, dirname))]
    sequence_list.sort()
    for sequence in sequence_list:
        constructor.beginInitializingSequence()
        constructor.setSequenceObjectCategory(sequence)
        constructor.setSequenceName('0')

        path = os.path.join(root_path, sequence)
        images = [image for image in os.listdir(path) if image.endswith('.jpg')]
        images.sort()

        current_index = 0
        for line in open(os.path.join(path, 'groundtruth.txt')):
            line = line.strip()
            values = [float(value) for value in line.split(',')]
            constructor.setFrameAttributes(constructor.addFrame(os.path.join(path, images[current_index])), values)
            current_index += 1

        constructor.endInitializingSequence()
