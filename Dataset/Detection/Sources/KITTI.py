from Dataset.DataSplit import DataSplit
import os


def construct_KITTI_Detection(constructor, seed):
    data_split = seed.data_split
    root_path = seed.root_path
    exclude_dontcare = seed.exclude_dontcare

    # https://github.com/NVIDIA/DIGITS/blob/master/digits/extensions/data/objectDetection/README.md
    constructor.setDatasetAttribute('annotation description',
                                    '''
#Values    Name      Description
----------------------------------------------------------------------------
   1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                     'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                     'Misc' or 'DontCare'
   1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
                     truncated refers to the object leaving image boundaries
   1    occluded     Integer (0,1,2,3) indicating occlusion state:
                     0 = fully visible, 1 = partly occluded
                     2 = largely occluded, 3 = unknown
   1    alpha        Observation angle of object, ranging [-pi..pi]
   4    bbox         2D bounding box of object in the image (0-based index):
                     contains left, top, right, bottom pixel coordinates
   3    dimensions   3D object dimensions: height, width, length (in meters)
   3    location     3D object location x,y,z in camera coordinates (in meters)
   1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
   1    score        Only for results: Float, indicating confidence in
                     detection, needed for p/r curves, higher is better.
''')

    assert data_split == DataSplit.Training

    annotation_root_path = os.path.join(root_path, '..', 'label_2')
    annotation_files = os.listdir(annotation_root_path)
    annotation_files = [annotation_file for annotation_file in annotation_files if annotation_file.endswith('.txt')]
    annotation_files.sort()

    for annotation_file in annotation_files:
        image_path = os.path.join(root_path, annotation_file[:-4] + '.png')
        constructor.beginInitializeImage()
        constructor.setImageName(annotation_file[:-4])
        constructor.setImagePath(image_path)
        annotation_file_path = os.path.join(annotation_root_path, annotation_file)
        for line in open(annotation_file_path, 'r', encoding='utf-8'):
            line = line.strip()
            if len(line) == 0:
                continue
            words = line.split(' ')
            assert len(words) == 15
            label = words[0]
            if exclude_dontcare and label == 'DontCare':
                continue
            truncated = float(words[1])
            occlusion = int(words[2])
            alpha = float(words[3])
            bounding_box = [float(words[4]), float(words[5]), float(words[6]), float(words[7])]
            bounding_box = [bounding_box[0], bounding_box[1], bounding_box[2] - bounding_box[0], bounding_box[3] - bounding_box[1]]
            dimensions = [float(words[8]), float(words[9]), float(words[10])]
            location = [float(words[11]), float(words[12]), float(words[13])]
            rotation_y = float(words[14])

            constructor.addObject(bounding_box, label, not (truncated > 0.8 or occlusion == 2),
                                  {'truncated': truncated, 'occlusion': occlusion, 'alpha': alpha,
                                   'dimensions': dimensions, 'location': location, 'rotation_y': rotation_y})
        constructor.endInitializeImage()
