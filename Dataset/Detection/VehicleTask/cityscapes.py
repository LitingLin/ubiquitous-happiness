from Dataset.Detection.VehicleTask.base import BaseDataset
from enum import Flag, auto
from Dataset.DataSplit import DataSplit
import os
import json

class CityscapesDataset(BaseDataset):
    class AnnotationSource(Flag):
        PreferFine = auto()
        FineOnly = auto()
        CoarseOnly = auto()

    def __init__(self, root_dir: str, data_type=DataSplit.Training | DataSplit.Validation, annotation_source=AnnotationSource.PreferFine):
        super(CityscapesDataset, self).__init__(root_dir, {
            0: 'person',
            1: 'rider',
            2: 'car',
            3: 'truck',
            4: 'bus',
            5: 'motorcycle',
            6: 'bicycle'
        })
        self.root_dir = root_dir
        self.accept_classes = {
            'person': 0, 'rider': 1,
            'car': 2, 'truck': 3, 'bus': 4, 'motorcycle': 5, 'bicycle': 6
        }
        annotation_paths = []

        if data_type & DataSplit.Training:
            if annotation_source == CityscapesDataset.AnnotationSource.PreferFine:
                annotation_paths.append(('gtFine', 'train'))
                annotation_paths.append(('gtCoarse', 'train_extra'))
            elif annotation_source == CityscapesDataset.AnnotationSource.FineOnly:
                annotation_paths.append(('gtFine', 'train'))
            elif annotation_source == CityscapesDataset.AnnotationSource.CoarseOnly:
                annotation_paths.append(('gtCoarse', 'train'))
                annotation_paths.append(('gtCoarse', 'train_extra'))
            else:
                raise Exception

        if data_type & DataSplit.Validation:
            if annotation_source == CityscapesDataset.AnnotationSource.PreferFine:
                annotation_paths.append(('gtFine', 'val'))
            elif annotation_source == CityscapesDataset.AnnotationSource.FineOnly:
                annotation_paths.append(('gtFine', 'val'))
            elif annotation_source == CityscapesDataset.AnnotationSource.CoarseOnly:
                annotation_paths.append(('gtCoarse', 'val'))
            else:
                raise Exception

        self._initialize(annotation_paths)

    def _initialize(self, annotation_paths):
        for annotation_path in annotation_paths:
            cities = os.listdir(os.path.join(self.root_dir, *annotation_path))
            for city in cities:
                files = os.listdir(os.path.join(self.root_dir, *annotation_path, city))
                files = [file for file in files if file.endswith('.json')]
                for json_file in files:
                    json_file_path = os.path.join(self.root_dir, *annotation_path, city, json_file)

                    with open(json_file_path, 'r') as fid:
                        file_content = fid.read()

                    json_objects = json.loads(file_content)
                    img_width = json_objects['imgWidth']
                    img_height = json_objects['imgHeight']

                    bounding_boxes = []
                    classes = []

                    for object in json_objects['objects']:
                        if object['label'] not in self.accept_classes:
                            continue

                        xmin = img_width
                        ymin = img_height
                        xmax = 0
                        ymax = 0

                        for x, y in object['polygon']:
                            if xmin > x:
                                xmin = x
                            if ymin > y:
                                ymin = y
                            if xmax < x:
                                xmax = x
                            if ymax < y:
                                ymax = y

                        bounding_box = (xmin, ymin, xmax-xmin, ymax-ymin)
                        class_ = self.accept_classes[object['label']]
                        bounding_boxes.append(bounding_box)
                        classes.append(class_)

                    if len(bounding_boxes) > 0:
                        file_name_parts = json_file[:-5].split('_')
                        if len(file_name_parts) != 5:
                            raise Exception
                        image_file_name = file_name_parts[0]+'_'+file_name_parts[1]+'_'+file_name_parts[2]+'_leftImg8bit.png'
                        image_file_relative_path = os.path.join('leftImg8bit', annotation_path[1], city, image_file_name)
                        if not os.path.exists(os.path.join(self.root_dir, image_file_relative_path)):
                            raise RuntimeError('Image file: {} not exists'.format(os.path.join(self.root_dir, image_file_relative_path)))
                        self.addRecord(image_file_relative_path, bounding_boxes, classes)
