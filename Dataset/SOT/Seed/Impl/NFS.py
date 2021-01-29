import os
from ..NFS import NFSDatasetVersionFlag
from Dataset.Type.data_split import DataSplit
from Dataset.SOT.Constructor.base import SingleObjectTrackingDatasetConstructor


_category_id_name_map = {0: 'airboard', 1: 'aircraft', 2: 'animal', 3: 'bag', 4: 'ball', 5: 'bicycle', 6: 'bird', 7: 'cup', 8: 'dollar', 9: 'drone', 10: 'face', 11: 'fish', 12: 'motorcycle', 13: 'person', 14: 'shuffleboard', 15: 'vehicle', 16: 'yoyo'}


def construct_NFS(constructor: SingleObjectTrackingDatasetConstructor, seed):
    assert seed.data_split == DataSplit.Full
    root_path = seed.root_path
    version = seed.nfs_version
    manual_anno_only = seed.manual_anno_only

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

    constructor.set_total_number_of_sequences(len(sequence_list))
    constructor.set_category_id_name_map(_category_id_name_map)
    category_name_id_map = {v: k for k, v in _category_id_name_map.items()}

    for sequence_name in sequence_list:
        sequence_images_path = os.path.join(root_path, sequence_name, subDirName, sequence_name)
        sequence_anno_file_path = os.path.join(root_path, sequence_name, subDirName, '{}.txt'.format(sequence_name))
        images = os.listdir(sequence_images_path)
        images = [image for image in images if image.endswith('.jpg')]
        images.sort()

        className = None
        track_id = None
        bounding_box_annotations = {}
        for line_count, line in enumerate(open(sequence_anno_file_path)):
            '''
            # https://github.com/cvondrick/vatic
            1   Track ID. All rows with the same ID belong to the same path.
            2   xmin. The top left x-coordinate of the bounding box.
            3   ymin. The top left y-coordinate of the bounding box.
            4   xmax. The bottom right x-coordinate of the bounding box.
            5   ymax. The bottom right y-coordinate of the bounding box.
            6   frame. The frame that this annotation represents.
            7   lost. If 1, the annotation is outside of the view screen.
            8   occluded. If 1, the annotation is occluded.
            9   generated. If 1, the annotation was automatically interpolated.
            10  label. The label for this annotation, enclosed in quotation marks.
            11+ attributes. Each column after this is an attribute.
            '''
            line = line.strip()
            first_quote_index = line.find('"')
            if first_quote_index == -1:
                raise Exception
            attributes = line[:first_quote_index].split()
            track_id_ = int(attributes[0])
            if track_id is None:
                track_id = track_id_
            else:
                assert track_id_ == track_id
            bbox = attributes[1:5]
            bbox = [int(v) for v in bbox]
            bbox[2] = bbox[2] - bbox[0]
            bbox[3] = bbox[3] - bbox[1]
            frame_index = int(attributes[5]) - 1
            if sequence_name == 'pingpong_2' and frame_index >= 263:
                continue
            if subDirName == '30':
                if frame_index % 8 != 0:
                    continue
                else:
                    frame_index /= 8
            out_of_view = bool(int(attributes[6]))
            occluded = bool(int(attributes[7]))
            generated = bool(int(attributes[8]))
            if generated and manual_anno_only:
                continue
            bounding_box_annotations[frame_index] = (bbox, out_of_view, occluded)
            second_quote_index = line.rfind('"')
            if second_quote_index == -1 or second_quote_index <= first_quote_index:
                raise Exception
            current_class_name = line[first_quote_index + 1: second_quote_index]
            if className is None:
                className = current_class_name
            elif className != current_class_name:
                raise Exception
        if len(bounding_box_annotations) == 0:
            continue
        assert className is not None
        with constructor.new_sequence(category_name_id_map[className]) as sequence_constructor:
            sequence_constructor.set_name(sequence_name)
            for index_of_image, image in enumerate(images):
                image_path = os.path.join(sequence_images_path, image)
                with sequence_constructor.new_frame() as frame_constructor:
                    frame_constructor.set_path(image_path)
                    if index_of_image in bounding_box_annotations:
                        bbox, out_of_view, occluded = bounding_box_annotations[index_of_image]
                        frame_constructor.set_bounding_box(bbox, validity=not(out_of_view or occluded))
                        frame_constructor.set_object_attribute('lost', out_of_view)
                        frame_constructor.set_object_attribute('occluded', occluded)
