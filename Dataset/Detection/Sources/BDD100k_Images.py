from Dataset.Detection.Base.constructor import DetectionDatasetConstructor
from Dataset.DataSplit import DataSplit
import json
import os


def construct_BDD100k_Images(constructor: DetectionDatasetConstructor, seed):
    root_path = seed.root_path
    data_split = seed.data_split
    labels_path = os.path.join(root_path, '..', '..', 'labels')

    auto_category_id_allocator = constructor.getAutoCategoryIdAllocationTool()

    def _construct(images_path: str, annotation_file_path: str):
        with open(annotation_file_path, 'r', encoding='utf-8') as fid:
            annotations = json.load(fid)

        for image_annotation in annotations:
            image_name = image_annotation['name']
            image_attributes = image_annotation['attributes']
            labels = image_annotation['labels']
            constructor.beginInitializeImage()
            constructor.setImagePath(os.path.join(images_path, image_name))
            constructor.setImageName(image_name[:-4])
            constructor.addImageAttributes(image_attributes)

            for label in labels:
                if 'box2d' not in label:
                    continue
                object_category = label['category']
                bounding_box = label['box2d']
                bounding_box = [bounding_box['x1'], bounding_box['y1'], bounding_box['x2'] - bounding_box['x1'], bounding_box['y2'] - bounding_box['y1']]
                object_attributes = label['attributes']
                occluded = object_attributes['occluded']
                truncated = object_attributes['truncated']
                constructor.addObject(bounding_box, auto_category_id_allocator.getOrAllocateCategoryId(object_category), not(occluded or truncated), object_attributes)
            constructor.endInitializeImage()

    if data_split & DataSplit.Training:
        _construct(os.path.join(root_path, 'train'), os.path.join(labels_path, 'bdd100k_labels_images_train.json'))
    if data_split & DataSplit.Validation:
        _construct(os.path.join(root_path, 'val'), os.path.join(labels_path, 'bdd100k_labels_images_val.json'))
