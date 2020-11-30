from Dataset.Detection.Base.constructor import DetectionDatasetConstructor
from Dataset.DataSplit import DataSplit
import os
import csv
from collections import namedtuple
from tqdm import tqdm


def construct_OpenImages(constructor: DetectionDatasetConstructor, seed):
    root_path = seed.root_path
    data_split = seed.data_split

    splits = []
    if data_split & DataSplit.Training:
        splits.append('train')
    if data_split & DataSplit.Validation:
        splits.append('validation')
    if data_split & DataSplit.Testing:
        splits.append('test')

    mid_name_mapper = {}

    for line in open(os.path.join(root_path, 'class-descriptions-boxable.csv'), 'r', encoding='utf-8'):
        line = line.strip()
        if len(line) == 0:
            continue
        words = line.split(',')
        assert len(words) == 2
        mid_name_mapper[words[0]] = words[1]

    def _construct_sub_dataset(images_path: str, annotation_file_path: str):
        with open(annotation_file_path, 'r', encoding='utf-8') as fid:
            csv_reader = csv.reader(fid)
            headings = next(csv_reader)
            Row = namedtuple('Row', headings)
            last_row_image = None
            last_image_size = None
            for r in tqdm(csv_reader):
                row = Row(*r)
                image_name = row.ImageID
                if last_row_image != image_name:
                    if last_row_image is not None:
                        constructor.endInitializeImage()
                    constructor.beginInitializeImage()
                    constructor.setImageName(image_name)
                    last_image_size = constructor.setImagePath(os.path.join(images_path, image_name + '.jpg'))
                    last_row_image = image_name
                object_category = mid_name_mapper[row.LabelName]
                bounding_box = [float(row.XMin) * last_image_size[0], float(row.XMax) * last_image_size[0], float(row.YMin) * last_image_size[1], float(row.YMax) * last_image_size[1]]
                bounding_box = [bounding_box[0], bounding_box[2], bounding_box[1] - bounding_box[0], bounding_box[3] - bounding_box[2]]

                constructor.addObject(bounding_box, object_category, attributes=
                                      {'IsOccluded': row.IsOccluded, 'IsTruncated': row.IsTruncated, 'IsGroupOf': row.IsGroupOf, 'IsDepiction': row.IsDepiction, 'IsInside': row.IsInside})
            constructor.endInitializeImage()
    if data_split & DataSplit.Training:
        _construct_sub_dataset(os.path.join(root_path, 'train'), os.path.join(root_path, 'train-annotations-bbox.csv'))
    if data_split & DataSplit.Validation:
        _construct_sub_dataset(os.path.join(root_path, 'validation'), os.path.join(root_path, 'validation-annotations-bbox.csv'))
    if data_split & DataSplit.Testing:
        _construct_sub_dataset(os.path.join(root_path, 'test'), os.path.join(root_path, 'test-annotations-bbox.csv'))
