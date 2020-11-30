from Dataset.Detection.VehicleTask.base import BaseDataset
import os
from Dataset.DataSplit import DataSplit
from typing import List, Tuple


class WebCamTDataset(BaseDataset):
    def __init__(self, root_dir, data_type=DataSplit.Training | DataSplit.Validation, include_passenger=False):
        class_names = {}
        for idx, name in enumerate(open(os.path.join(root_dir, 'city_cam.names'))):
            class_names[idx] = name
        if include_passenger:
            class_names[len(class_names)] = 'passenger'

        super(WebCamTDataset, self).__init__(root_dir, class_names)

        def _parseTrainTestSeparationFile(path):
            paths = []
            for line in open(path, 'r'):
                line = line.strip()
                if len(line) > 0:
                    words = line.split('-')
                    paths.append((words[0], line))
            return paths

        train_paths = []
        val_paths = []

        train_paths += _parseTrainTestSeparationFile(
            os.path.join(root_dir, 'train_test_separation', 'Downtown_Train.txt'))
        train_paths += _parseTrainTestSeparationFile(
            os.path.join(root_dir, 'train_test_separation', 'Parkway_Train.txt'))

        val_paths += _parseTrainTestSeparationFile(
            os.path.join(root_dir, 'train_test_separation', 'Downtown_Test.txt'))
        val_paths += _parseTrainTestSeparationFile(
            os.path.join(root_dir, 'train_test_separation', 'Parkway_Test.txt'))

        paths = []
        if data_type & DataSplit.Training:
            paths += train_paths
        if data_type & DataSplit.Validation:
            paths += val_paths

        self._initialize(paths, include_passenger=include_passenger)

    def _initialize(self, paths: List[Tuple[str, str]], include_passenger: bool):
        for path in paths:
            xml_files = os.listdir(os.path.join(self.root_dir, *path))
            xml_files = [file_name for file_name in xml_files if file_name.endswith('.xml')]
            for xml_file in xml_files:
                with open(os.path.join(self.root_dir, *path, xml_file), 'r') as fid:
                    file_content = fid.read()

                bounding_boxes = []
                categories = []

                image_file_name = xml_file[:-3] + 'jpg'
                image_relative_path = os.path.join(*path, image_file_name)
                if not os.path.exists(os.path.join(self.root_dir, image_relative_path)):
                    print('Image file: {} not exists'.format(image_relative_path))
                    continue

                begin_index = 0
                image_size = self._getWidthAndHeight(file_content, begin_index)
                if image_size is None:
                    print('Invalid annotation file: {}'.format(os.path.join(*path, xml_file)))
                    continue

                width, height, begin_index = image_size

                while True:
                    vehicle = self._findNextObject(file_content, begin_index)
                    if vehicle is None:
                        break

                    is_vehicle = vehicle[0]
                    category = vehicle[1]
                    x, y, w, h = vehicle[2:6]
                    begin_index = vehicle[6]

                    if not include_passenger:
                        if not is_vehicle:
                            continue

                    def _printInvalidBoundingBoxAndFixed():
                        print('Invalid bounding box in annotation file: {}. Fixed.'.format(os.path.join(*path, xml_file)))
                        print((x, y, w, h))

                    def _printInvalidBoundingBox():
                        print('Invalid bounding box in annotation file: {}.'.format(os.path.join(*path, xml_file)))
                        print((x, y, w, h))

                    if w <= 0 or h <= 0:
                        _printInvalidBoundingBox()
                        continue

                    if x < 0:
                        _printInvalidBoundingBoxAndFixed()
                        x = 0

                    if y < 0:
                        _printInvalidBoundingBoxAndFixed()
                        y = 0

                    if x + w >= width:
                        _printInvalidBoundingBoxAndFixed()
                        w = width - x - 1

                    if y + h >= height:
                        _printInvalidBoundingBoxAndFixed()
                        h = height - y - 1

                    if is_vehicle:
                        if category < 0 or category >= len(self.classNamesMapper) - 1:
                            print('Invalid type in annotation file: {}'.format(os.path.join(*path, xml_file)))
                            continue

                    bounding_boxes.append((x, y, w, h))
                    categories.append(category)

                if len(bounding_boxes) > 0:
                    self.addRecord(image_relative_path, bounding_boxes, categories)

    def _findNextObject(self, file_content: str, begin_index: int):
        vehicle_begin = file_content.find('<vehicle>', begin_index)
        passenger_begin = file_content.find('<passengers>', begin_index)

        if vehicle_begin == -1 and passenger_begin == -1:
            return None

        if vehicle_begin == -1:
            is_vehicle = False
        elif passenger_begin == -1:
            is_vehicle = True
        elif vehicle_begin < passenger_begin:
            is_vehicle = True
        else:
            is_vehicle = False

        if is_vehicle:
            begin_index = vehicle_begin + 9
        else:
            begin_index = passenger_begin + 12

        xmax_begin_index = file_content.find('<xmax>', begin_index)
        xmax_end_index = file_content.find('</xmax>', xmax_begin_index)
        xmin_begin_index = file_content.find('<xmin>', xmax_end_index)

        xmin_end_index = file_content.find('</xmin>', xmin_begin_index)
        ymax_begin_index = file_content.find('<ymax>', xmin_end_index)
        ymax_end_index = file_content.find('</ymax>', ymax_begin_index)
        ymin_begin_index = file_content.find('<ymin>', ymax_end_index)
        ymin_end_index = file_content.find('</ymin>', ymin_begin_index)
        if is_vehicle:
            type_begin_index = file_content.find('<type>', ymin_end_index)
            type_end_index = file_content.find('</type>', type_begin_index)
            end_index = type_end_index + 7
        else:
            end_index = ymin_end_index + 7

        xmax = int(file_content[xmax_begin_index + 6: xmax_end_index])
        xmin = int(file_content[xmin_begin_index + 6: xmin_end_index])
        ymax = int(file_content[ymax_begin_index + 6: ymax_end_index])
        ymin = int(file_content[ymin_begin_index + 6: ymin_end_index])
        if is_vehicle:
            object_category = int(file_content[type_begin_index + 6: type_end_index]) - 1
        else:
            object_category = len(self.classNamesMapper) - 1

        x = xmin
        y = ymin
        w = xmax - xmin
        h = ymax - ymin

        return is_vehicle, object_category, x, y, w, h, end_index


    @staticmethod
    def _getWidthAndHeight(file_content: str, begin_index: int):
        width_begin_index = file_content.find('<width>', begin_index)
        if width_begin_index == -1:
            return None
        width_end_index = file_content.find('</width>', width_begin_index)
        height_begin_index = file_content.find('<height>', width_end_index)

        height_end_index = file_content.find('</height>', height_begin_index)

        width = int(file_content[width_begin_index + 7: width_end_index])
        height = int(file_content[height_begin_index + 8: height_end_index])

        return width, height, height_end_index + 9
