from Dataset.Detection.VehicleTask.base import BaseDataset
from Dataset.DataSplit import DataSplit
import os
import json
from typing import List, Dict, Tuple


class COCO2014Dataset(BaseDataset):
    def __init__(self, root_path: str, data_type=DataSplit.Training | DataSplit.Validation):
        super(COCO2014Dataset, self).__init__(root_path, {
            0: 'person',
            1: 'bicycle',
            2: 'car',
            3: 'motorcycle',
            4: 'bus',
            5: 'truck'
        })
        self.accept_classes = {
            1: 0,
            2: 1,
            3: 2,
            4: 3,
            6: 4,
            8: 5
        }
        annotation_paths = []
        if data_type & DataSplit.Training:
            annotation_paths.append(('annotations', 'instances_train2014.json'))
        if data_type & DataSplit.Validation:
            annotation_paths.append(('annotations', 'instances_val2014.json'))

        self._initialize(annotation_paths)

    def _initialize(self, annotation_paths):
        images: Dict[int: Tuple[str, List[Tuple[float, float, float, float]], List[int]]]={}

        for annotation_path in annotation_paths:
            if 'train' in annotation_path[1]:
                image_folder_name = 'train2014'
            elif 'val' in annotation_path[1]:
                image_folder_name = 'val2014'
            else:
                raise Exception
            with open(os.path.join(self.root_dir, *annotation_path), 'rb') as fid:
                json_objects = json.load(fid)

            for annotation in json_objects['annotations']:
                category_id = annotation['category_id']
                if category_id not in self.accept_classes:
                    continue

                image_id = annotation['image_id']
                if image_id not in images:
                    images[image_id] = ['', [], []]
                bbox = tuple(annotation['bbox'])
                images[image_id][1].append(bbox)
                images[image_id][2].append(self.accept_classes[category_id])

            for image in json_objects['images']:
                image_id = image['id']
                if image_id not in images:
                    continue
                file_name = image['file_name']
                images[image_id][0] = os.path.join(image_folder_name, file_name)
                if not os.path.exists(os.path.join(self.root_dir, image_folder_name, file_name)):
                    raise Exception

        for image_info in images.values():
            self.addRecord(image_info[0], image_info[1], image_info[2])
