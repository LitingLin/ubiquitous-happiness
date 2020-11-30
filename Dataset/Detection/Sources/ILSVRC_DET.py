from Dataset.DataSplit import DataSplit
import os
from Dataset.imagenet_200 import ImageNet200


def construct_ILSVRC_DET(constructor, seed):
    imagenet_id_name_map = {id_: name for id_, name in zip(ImageNet200.wn_id, ImageNet200.names)}
    root_path = seed.root_path
    data_split = seed.data_split

    annotation_path = os.path.join(root_path, 'Annotations', 'DET')
    annotation_paths = []
    image_paths = []

    if data_split & DataSplit.Training:
        path = os.path.join(annotation_path, 'train')
        for dirpath, _, filelist in os.walk(path):
            if len(filelist) > 0:
                annotation_paths.append(dirpath)
                image_paths.append(dirpath.replace('Annotations', 'Data', 1))
    elif data_split & DataSplit.Validation:
        annotation_paths.append(os.path.join('Annotations', 'DET', 'val'))
        image_paths.append(os.path.join('Data', 'DET', 'val'))

    for annotation_path, image_path in zip(annotation_paths, image_paths):
        annotation_file_names = os.listdir(os.path.join(root_path, annotation_path))
        for annotation_file_name in annotation_file_names:
            annotation_file = os.path.join(root_path, annotation_path, annotation_file_name)
            with open(annotation_file, 'r') as fid:
                file_content = fid.read()

            offset = 0
            bounding_boxes = []
            object_classes = []

            def _findNextObject(file_content: str, begin_index: int):
                object_name_begin_index = file_content.find('<name>', begin_index)
                if object_name_begin_index == -1:
                    return None
                object_name_end_index = file_content.find('</name>', object_name_begin_index)
                xmin_begin_index = file_content.find('<xmin>', object_name_end_index)
                xmin_end_index = file_content.find('</xmin>', xmin_begin_index)
                xmax_begin_index = file_content.find('<xmax>', xmin_end_index)
                xmax_end_index = file_content.find('</xmax>', xmax_begin_index)
                ymin_begin_index = file_content.find('<ymin>', xmax_end_index)
                ymin_end_index = file_content.find('</ymin>', ymin_begin_index)
                ymax_begin_index = file_content.find('<ymax>', ymin_end_index)
                ymax_end_index = file_content.find('</ymax>', ymax_begin_index)

                object_name = file_content[object_name_begin_index + len('<name>'): object_name_end_index]
                object_name = imagenet_id_name_map[object_name]

                xmax = int(file_content[xmax_begin_index + 6: xmax_end_index])
                xmin = int(file_content[xmin_begin_index + 6: xmin_end_index])
                ymax = int(file_content[ymax_begin_index + 6: ymax_end_index])
                ymin = int(file_content[ymin_begin_index + 6: ymin_end_index])

                x = xmin
                y = ymin
                w = xmax - xmin
                h = ymax - ymin

                return object_name, x, y, w, h, ymax_end_index + 7

            while True:
                next_object_in_annotation_file = _findNextObject(file_content, offset)
                if next_object_in_annotation_file is None:
                    break
                object_name, x, y, w, h, offset = next_object_in_annotation_file
                bounding_boxes.append((x, y, w, h))
                object_classes.append(object_name)
            if len(bounding_boxes) > 0:
                image_name = annotation_file_name[:annotation_file_name.rfind('.')]
                current_image = os.path.join(root_path, image_path, image_name + '.JPEG')
                constructor.beginInitializeImage()
                constructor.setImageName(image_name)
                constructor.setImagePath(current_image)
                for bounding_box, object_category in zip(bounding_boxes, object_classes):
                    constructor.addObject(bounding_box, object_category)
                constructor.endInitializeImage()
