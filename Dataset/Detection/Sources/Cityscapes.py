from Dataset.Detection.FactorySeeds.Cityscapes import CityscapesAnnotationSource
from Dataset.MetaInformation.cityscapes import name2label
from Dataset.DataSplit import DataSplit
import os
import json


def construct_CityScapes(constructor, seed):
    root_path = seed.root_path
    data_split = seed.data_split
    annotation_source = seed.annotation_source
    things_only = seed.things_only

    annotation_paths = []

    if data_split & DataSplit.Training:
        if annotation_source == CityscapesAnnotationSource.PreferFine:
            annotation_paths.append(('gtFine', 'train'))
            annotation_paths.append(('gtCoarse', 'train_extra'))
        elif annotation_source == CityscapesAnnotationSource.FineOnly:
            annotation_paths.append(('gtFine', 'train'))
        elif annotation_source == CityscapesAnnotationSource.CoarseOnly:
            annotation_paths.append(('gtCoarse', 'train'))
            annotation_paths.append(('gtCoarse', 'train_extra'))
        else:
            raise Exception

    if data_split & DataSplit.Validation:
        if annotation_source == CityscapesAnnotationSource.PreferFine:
            annotation_paths.append(('gtFine', 'val'))
        elif annotation_source == CityscapesAnnotationSource.FineOnly:
            annotation_paths.append(('gtFine', 'val'))
        elif annotation_source == CityscapesAnnotationSource.CoarseOnly:
            annotation_paths.append(('gtCoarse', 'val'))
        else:
            raise Exception

    for annotation_path in annotation_paths:
        cities = os.listdir(os.path.join(root_path, *annotation_path))
        for city in cities:
            files = os.listdir(os.path.join(root_path, *annotation_path, city))
            files = [file for file in files if file.endswith('.json')]
            for json_file in files:
                json_file_path = os.path.join(root_path, *annotation_path, city, json_file)

                with open(json_file_path, 'r') as fid:
                    file_content = fid.read()

                json_objects = json.loads(file_content)
                img_width = json_objects['imgWidth']
                img_height = json_objects['imgHeight']

                bounding_boxes = []
                classes = []

                for object in json_objects['objects']:
                    class_ = object['label']
                    if things_only:
                        if class_ not in name2label:
                            continue
                        if not name2label[class_].hasInstances:
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

                    bounding_box = [xmin, ymin, xmax-xmin, ymax-ymin]
                    bounding_boxes.append(bounding_box)
                    classes.append(class_)

                if len(bounding_boxes) > 0:
                    file_name_parts = json_file[:-5].split('_')
                    if len(file_name_parts) != 5:
                        raise Exception
                    image_file_name = file_name_parts[0]+'_'+file_name_parts[1]+'_'+file_name_parts[2]+'_leftImg8bit.png'
                    if image_file_name == 'troisdorf_000000_000073_leftImg8bit.png':
                        continue
                    image_file_path = os.path.join(root_path, 'leftImg8bit', annotation_path[1], city, image_file_name)
                    constructor.beginInitializeImage()
                    constructor.setImageName(image_file_name)
                    constructor.setImagePath(image_file_path)
                    for bounding_box, class_ in zip(bounding_boxes, classes):
                        constructor.addObject(bounding_box, class_)
                    constructor.endInitializeImage()
