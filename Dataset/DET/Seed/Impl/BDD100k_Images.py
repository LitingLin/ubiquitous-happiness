from Dataset.DET.Constructor.base import DetectionDatasetConstructor
from Dataset.Type.data_split import DataSplit
import json
import os
from data.types.bounding_box_format import BoundingBoxFormat

_category_id_name_map = {0: 'bike', 1: 'bus', 2: 'car', 3: 'motor', 4: 'person', 5: 'rider', 6: 'traffic light', 7: 'traffic sign', 8: 'train', 9: 'truck'}


def construct_BDD100k_Images(constructor: DetectionDatasetConstructor, seed):
    root_path = seed.root_path
    data_split = seed.data_split
    labels_path = os.path.join(root_path, '..', '..', 'labels')

    category_name_id_map = {v: k for k, v in _category_id_name_map.items()}
    constructor.set_category_id_name_map(_category_id_name_map)
    constructor.set_bounding_box_format(BoundingBoxFormat.XYXY)

    def _construct(images_path: str, annotation_file_path: str):
        with open(annotation_file_path, 'r', encoding='utf-8') as fid:
            annotations = json.load(fid)

        constructor.set_total_number_of_images(len(annotations))
        for image_annotation in annotations:
            image_name = image_annotation['name']
            image_attributes = image_annotation['attributes']
            labels = image_annotation['labels']
            with constructor.new_image() as image_constructor:
                image_constructor.set_path(os.path.join(images_path, image_name))
                image_constructor.merge_attributes(image_attributes)

                for label in labels:
                    if 'box2d' not in label:
                        continue
                    object_category = label['category']
                    bounding_box = label['box2d']
                    bounding_box = [bounding_box['x1'], bounding_box['y1'], bounding_box['x2'], bounding_box['y2']]
                    object_attributes = label['attributes']
                    occluded = object_attributes['occluded']
                    truncated = object_attributes['truncated']
                    with image_constructor.new_object() as object_constructor:
                        object_constructor.set_bounding_box(bounding_box, validity=not(occluded or truncated))
                        object_constructor.set_category_id(category_name_id_map[object_category])
                        object_constructor.merge_attributes(object_attributes)

    if data_split & DataSplit.Training:
        _construct(os.path.join(root_path, 'train'), os.path.join(labels_path, 'bdd100k_labels_images_train.json'))
    if data_split & DataSplit.Validation:
        _construct(os.path.join(root_path, 'val'), os.path.join(labels_path, 'bdd100k_labels_images_val.json'))
