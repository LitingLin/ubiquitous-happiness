from Dataset.Detection.VehicleTask.base import BaseDataset

import os
import subprocess
from typing import Dict, Tuple, List
from enum import IntFlag, auto


class QingDaoDataset(BaseDataset):
    class SceneAttribute(IntFlag):
        DianJing = auto()
        LuKou = auto()
        GaoDian = auto()

    def __init__(self, root_dir, scene_types=SceneAttribute.DianJing|SceneAttribute.GaoDian|SceneAttribute.LuKou):
        self.occludeds = []
        self.outOfViews = []

        labels = None
        videos_types = os.listdir(root_dir)
        for video_type in videos_types:
            if not os.path.isdir(os.path.join(root_dir, video_type)):
                continue

            if video_type == 'DianJing':
                if not scene_types & QingDaoDataset.SceneAttribute.DianJing:
                    continue
            elif video_type == 'LuKou':
                if not scene_types & QingDaoDataset.SceneAttribute.LuKou:
                    continue
            elif video_type == 'GaoDian':
                if not scene_types & QingDaoDataset.SceneAttribute.GaoDian:
                    continue
            else:
                continue

            videos = os.listdir(os.path.join(root_dir, video_type, 'Video'))

            annotation_types = os.listdir(os.path.join(root_dir, video_type, 'Result'))
            assert len(annotation_types) == 2
            annotation_folder = None
            for annotation_type in annotation_types:
                if not annotation_type.endswith('AreaAnnotation'):
                    annotation_folder = os.path.join(os.path.join(root_dir, video_type, 'Result', annotation_type))

            assert annotation_folder is not None

            for video in videos:
                if not video.endswith('.mp4'):
                    continue

                frame_attributes: Dict[int, Tuple[List[int], List[Tuple[float, float, float, float]], List[int], List[int]]] = {}

                video_path = os.path.join(root_dir, video_type, 'Video', video)
                output_path = os.path.join(root_dir, video_type, 'Video', video[:-4])
                if not os.path.exists(output_path):
                    os.mkdir(output_path)
                    subprocess.check_call(['ffmpeg', '-i', video_path, os.path.join(output_path, '%04d.png')])

                annotation_path = os.path.join(annotation_folder, video[:-3] + 'xml')
                with open(annotation_path, 'rb') as fid:
                    annotation_file_content = fid.read().decode("utf-8")

                begin_index = 0
                current_file_labels, begin_index = self._getLabelNames(annotation_file_content, begin_index)

                if labels is not None:
                    assert current_file_labels == labels
                else:
                    labels = current_file_labels
                    label_index_name_mapper = {}
                    label_name_index_mapper = {}
                    for index, label in enumerate(labels):
                        label_index_name_mapper[index] = label
                        label_name_index_mapper[label] = index
                    super(QingDaoDataset, self).__init__(root_dir, label_index_name_mapper)

                while True:
                    attributes = self._getNextObjectBoundingBoxes(annotation_file_content, begin_index)
                    if attributes is None:
                        break

                    (frame_indices, bounding_boxes, categories, occludeds, out_of_views), begin_index = attributes

                    for frame_index, bounding_box, category, occluded, out_of_view in zip(frame_indices, bounding_boxes, categories, occludeds, out_of_views):
                        if frame_index not in frame_attributes:
                            frame_attributes[frame_index] = ([], [], [], [])

                        attribute = frame_attributes[frame_index]
                        attribute[0].append(category)
                        attribute[1].append(bounding_box)
                        attribute[2].append(occluded)
                        attribute[3].append(out_of_view)

                for frame_index, (categories, bounding_boxes, occludeds, out_of_views) in frame_attributes.items():
                    frame_relative_path = os.path.join(output_path, '{:0>4d}.png'.format(frame_index + 1))
                    assert os.path.exists(os.path.join(root_dir, frame_relative_path))
                    categories = [label_name_index_mapper[category] for category in categories]
                    self.addRecord(frame_relative_path, bounding_boxes, categories)
                    self.occludeds.append(occludeds)
                    self.outOfViews.append(out_of_views)

    @staticmethod
    def _getLabelNames(file_content: str, begin_index: int):
        begin_index = file_content.find('<name>vehicle</name>', begin_index)
        assert begin_index != -1
        begin_feature_string = '<attribute>@select=type:'
        begin_index = file_content.find(begin_feature_string, begin_index)
        assert begin_index != -1
        begin_index += len(begin_feature_string)
        end_feature_string = '</attribute>'
        end_index = file_content.find(end_feature_string, begin_index)
        labels_string = file_content[begin_index: end_index]
        return labels_string.split(','), end_index + len(end_feature_string)

    @staticmethod
    def _getNextObjectBoundingBoxes(file_content: str, begin_index: int):
        track_begin_string = '<track'
        track_end_string = '</track>'

        track_begin_index = file_content.find(track_begin_string, begin_index)
        if begin_index == -1:
            return None

        track_end_index = file_content.find(track_end_string, begin_index) + len(track_end_string)

        boundingBoxes = []
        catelogies = []
        frame_indices = []
        outOfViews = []
        occludeds = []

        begin_index = track_begin_index + len(track_begin_string)

        while True:
            feature_string = '<box'
            begin_index = file_content.find(feature_string, begin_index)
            if begin_index == -1:
                break
            if begin_index >= track_end_index:
                break

            begin_index += len(feature_string)

            def getAttribute(string, begin_feature_string, end_feature_string, begin_index):
                begin_index = string.find(begin_feature_string, begin_index)
                begin_index += len(begin_feature_string)
                end_index = string.find(end_feature_string, begin_index)
                return string[begin_index: end_index], end_index + len(end_feature_string)

            frame_index, begin_index = getAttribute(file_content, 'frame="', '"', begin_index)
            frame_index = int(frame_index)

            xmin, begin_index = getAttribute(file_content, 'xtl="', '"', begin_index)
            xmin = float(xmin)

            ymin, begin_index = getAttribute(file_content, 'ytl="', '"', begin_index)
            ymin = float(ymin)

            xmax, begin_index = getAttribute(file_content, 'xbr="', '"', begin_index)
            xmax = float(xmax)

            ymax, begin_index = getAttribute(file_content, 'ybr="', '"', begin_index)
            ymax = float(ymax)

            outOfView, begin_index = getAttribute(file_content, 'outside="', '"', begin_index)
            outOfView = int(outOfView)

            occluded, begin_index = getAttribute(file_content, 'occluded="', '"', begin_index)
            occluded = int(occluded)

            label, begin_index = getAttribute(file_content, '<attribute name="type">', '</attribute>', begin_index)

            boundingBoxes.append((xmin, ymin, xmax - xmin, ymax - ymin))
            catelogies.append(label)
            frame_indices.append(frame_index)
            outOfViews.append(outOfView)
            occludeds.append(occluded)

        if len(boundingBoxes) == 0:
            return None
        return (frame_indices, boundingBoxes, catelogies, occludeds, outOfViews), track_end_index
