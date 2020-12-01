from Dataset.Detection.FactorySeeds.COCO import COCOVersion
from Dataset.DataSplit import DataSplit
import os
import json
from typing import List, Dict, Tuple


def construct_COCO(constructor, seed):
    root_path = seed.root_path
    include_crowd = seed.include_crowd
    data_split = seed.data_split
    version = seed.version

    images: Dict[int: Tuple[str, List[Tuple[float, float, float, float]], List[int]]] = {}

    def _construct(image_folder_name: str, annotation_file_name: str):
        with open(os.path.join(root_path, 'annotations', annotation_file_name), 'r', encoding='utf-8') as fid:
            json_objects = json.load(fid)

        categories_ = json_objects['categories']
        categories = {}
        for category in categories_:
            categories[category['id']] = {'name': category['name'], 'supercategory': category['supercategory']}

        for annotation in json_objects['annotations']:
            if not include_crowd:
                if annotation['iscrowd'] != 0:
                    continue
            category_id = annotation['category_id']
            image_id = annotation['image_id']
            if image_id not in images:
                images[image_id] = ['', '', [], [], []]
            bbox = list(annotation['bbox'])
            images[image_id][2].append(bbox)
            images[image_id][3].append(categories[category_id]['name'])
            if include_crowd:
                images[image_id][4].append({'iscrowd': annotation['iscrowd'] > 0})

        for image in json_objects['images']:
            image_id = image['id']
            if image_id not in images:
                continue
            file_name = image['file_name']
            images[image_id][0] = os.path.join('images', image_folder_name, file_name)
            images[image_id][1] = file_name

        for image_info in images.values():
            constructor.beginInitializeImage()
            constructor.setImagePath(os.path.join(root_path, image_info[0]))
            constructor.setImageName(image_info[1])
            if include_crowd:
                for bounding_box, category_name, attributes in zip(image_info[2], image_info[3], image_info[4]):
                    constructor.addObject(bounding_box, category_name, attributes=attributes)
            else:
                for bounding_box, category_name in zip(image_info[2], image_info[3]):
                    constructor.addObject(bounding_box, category_name)
            constructor.endInitializeImage()

    if version == COCOVersion._2014:
        if data_split & DataSplit.Training:
            _construct('train2014', 'instances_train2014.json')
        if data_split & DataSplit.Validation:
            _construct('val2014', 'instances_val2014.json')
    elif version == COCOVersion._2017:
        if data_split & DataSplit.Training:
            _construct('train2017', 'instances_train2017.json')
        if data_split & DataSplit.Validation:
            _construct('val2017', 'instances_val2017.json')
    else:
        raise Exception
