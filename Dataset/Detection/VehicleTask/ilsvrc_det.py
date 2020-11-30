from Dataset.DataSplit import DataSplit
from Dataset.Detection.VehicleTask.base import BaseDataset
import os
from typing import List


class ILSVRC_DET_Dataset(BaseDataset):
    def __init__(self, root_dir: str, data_type=DataSplit.Training | DataSplit.Validation):
        super(ILSVRC_DET_Dataset, self).__init__(root_dir, {
            0: 'bicycle',
            1: 'bus',
            2: 'car',
            3: 'motorcycle',
            4: 'person'
        })
        self.accept_classes = {
            'n02834778': 0,
            'n02924116': 1,
            'n02958343': 2,
            'n03790512': 3,
            'n00007846': 4,
            # 'n04509417': 'unicycle'
        }

        annotation_path = os.path.join(root_dir, 'Annotations', 'DET')
        annotation_paths = []
        image_paths = []

        if data_type & DataSplit.Training:
            path = os.path.join(annotation_path, 'train')
            for dirpath, _, filelist in os.walk(path):
                if len(filelist) > 0:
                    if root_dir.endswith('/') or root_dir.endswith('\\'):
                        dirpath = dirpath[len(root_dir):]
                    else:
                        dirpath = dirpath[len(root_dir)+1:]

                    annotation_paths.append(dirpath)
                    image_paths.append(dirpath.replace('Annotations', 'Data', 1))
        elif data_type & DataSplit.Validation:
            annotation_paths.append(os.path.join('Annotations', 'DET', 'val'))
            image_paths.append(os.path.join('Data', 'DET', 'val'))

        self._initialize(annotation_paths, image_paths)

    def _initialize(self, annotation_paths: List[str], image_paths: List[str]):
        for annotation_path, image_path in zip(annotation_paths, image_paths):
            annotation_file_names = os.listdir(os.path.join(self.root_dir, annotation_path))
            for annotation_file_name in annotation_file_names:
                annotation_file = os.path.join(self.root_dir, annotation_path, annotation_file_name)
                with open(annotation_file, 'r') as fid:
                    file_content = fid.read()

                offset = 0
                bounding_boxes = []
                object_classes = []
                while True:
                    next_object_in_annotation_file = self._findNextObject(file_content, offset)
                    if next_object_in_annotation_file is None:
                        break
                    object_name, x, y, w, h, offset = next_object_in_annotation_file
                    if object_name in self.accept_classes:
                        bounding_boxes.append((x, y, w, h))
                        object_classes.append(self.accept_classes[object_name])
                if len(bounding_boxes) > 0:
                    current_image = os.path.join(image_path, annotation_file_name[:annotation_file_name.rfind('.')] + '.JPEG')
                    if not os.path.exists(os.path.join(self.root_dir, current_image)):
                        raise Exception('image not exists: {}'.format(current_image))
                    self.addRecord(current_image, bounding_boxes, object_classes)

    @staticmethod
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

        xmax = int(file_content[xmax_begin_index + 6: xmax_end_index])
        xmin = int(file_content[xmin_begin_index + 6: xmin_end_index])
        ymax = int(file_content[ymax_begin_index + 6: ymax_end_index])
        ymin = int(file_content[ymin_begin_index + 6: ymin_end_index])

        x = xmin
        y = ymin
        w = xmax - xmin
        h = ymax - ymin

        return object_name, x, y, w, h, ymax_end_index + 7
