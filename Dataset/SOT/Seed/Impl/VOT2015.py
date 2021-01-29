import os
from Dataset.SOT.Constructor.base import SingleObjectTrackingDatasetConstructor


def constructVOT2015(constructor: SingleObjectTrackingDatasetConstructor, seed):
    root_path = seed.root_path
    sequence_list = os.listdir(root_path)
    sequence_list = [dirname for dirname in sequence_list if os.path.isdir(os.path.join(root_path, dirname))]
    sequence_list.sort()
    constructor.set_total_number_of_sequences(len(sequence_list))
    for sequence in sequence_list:
        with constructor.new_sequence() as sequence_constructor:
            sequence_constructor.set_name(sequence)
            path = os.path.join(root_path, sequence)
            images = [image for image in os.listdir(path) if image.endswith('.jpg')]
            images.sort()

            current_index = 0
            for line in open(os.path.join(path, 'groundtruth.txt')):
                line = line.strip()
                values = [float(value) for value in line.split(',')]
                with sequence_constructor.new_frame() as frame_constructor:
                    frame_constructor.set_path(os.path.join(path, images[current_index]))
                    frame_constructor.set_bounding_box(values)
                current_index += 1
