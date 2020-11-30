from Dataset.MOT.Base.constructor import MultipleObjectTrackingDatasetConstructor
from Dataset.MOT.FactorySeeds.QingDao import QingDaoDataset_SceneTypes
import os
import subprocess


def construct_QingDao(constructor: MultipleObjectTrackingDatasetConstructor, seed):
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

    def _getNextObjectBoundingBoxes(file_content: str, begin_index: int):
        track_begin_string = '<track'
        track_end_string = '</track>'

        track_begin_index = file_content.find(track_begin_string, begin_index)
        if track_begin_index == -1:
            return None

        begin_index = track_begin_index + len(track_begin_string)

        track_end_index = file_content.find(track_end_string, begin_index) + len(track_end_string)

        boundingBoxes = []
        category = None
        frame_indices = []
        outOfViews = []
        occludeds = []


        track_id_begin_string = 'id="'
        track_id_end_string = '" label'
        begin_index = file_content.find(track_id_begin_string, begin_index)
        begin_index += len(track_id_begin_string)
        track_id_end_index = file_content.find(track_id_end_string, begin_index)
        track_id = int(file_content[begin_index: track_id_end_index])
        begin_index = track_id_end_index + len(track_id_end_string)

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
            outOfView = bool(int(outOfView))

            occluded, begin_index = getAttribute(file_content, 'occluded="', '"', begin_index)
            occluded = bool(int(occluded))

            label, begin_index = getAttribute(file_content, '<attribute name="type">', '</attribute>', begin_index)
            if category is None:
                category = label
            else:
                assert category == label

            boundingBoxes.append([xmin, ymin, xmax - xmin, ymax - ymin])
            frame_indices.append(frame_index)
            outOfViews.append(outOfView)
            occludeds.append(occluded)

        if len(boundingBoxes) == 0:
            return None
        return track_id, category, (frame_indices, boundingBoxes, occludeds, outOfViews), track_end_index

    root_path = seed.root_path
    dataset_scene_type = seed.scene_type

    scene_types = os.listdir(root_path)
    for scene_type in scene_types:
        scene_path = os.path.join(root_path, scene_type)
        if not os.path.isdir(os.path.join(root_path, scene_type)):
            continue

        if scene_type == 'DianJing':
            if not dataset_scene_type & QingDaoDataset_SceneTypes.DianJing:
                continue
        elif scene_type == 'LuKou':
            if not dataset_scene_type & QingDaoDataset_SceneTypes.LuKou:
                continue
        elif scene_type == 'GaoDian':
            if not dataset_scene_type & QingDaoDataset_SceneTypes.GaoDian:
                continue
        else:
            continue

        videos = os.listdir(os.path.join(scene_path, 'Video'))

        annotation_types = os.listdir(os.path.join(scene_path, 'Result'))
        assert len(annotation_types) == 2
        annotation_folder = None
        for annotation_type in annotation_types:
            if not annotation_type.endswith('AreaAnnotation'):
                annotation_folder = os.path.join(os.path.join(scene_path, 'Result', annotation_type))

        assert annotation_folder is not None

        for video in videos:
            if not video.endswith('.mp4'):
                continue

            video_path = os.path.join(scene_path, 'Video', video)
            video_name = video[:-4]
            output_path = os.path.join(scene_path, 'Video', video_name)
            if not os.path.exists(output_path):
                os.mkdir(output_path)
                subprocess.check_call(['ffmpeg', '-i', video_path, os.path.join(output_path, '%04d.png')])

            constructor.beginInitializingSequence()
            constructor.setSequenceName(video_name)
            image_files = os.listdir(output_path)
            image_files.sort()
            for image_file in image_files:
                constructor.addFrame(os.path.join(output_path, image_file))

            annotation_path = os.path.join(annotation_folder, video_name + '.xml')
            with open(annotation_path, 'rb') as fid:
                annotation_file_content = fid.read().decode("utf-8")

            object_id_category_mapper = {}

            begin_index = 0
            while True:
                attributes = _getNextObjectBoundingBoxes(annotation_file_content, begin_index)
                if attributes is None:
                    break

                track_id, category, (frame_indices, boundingBoxes, occludeds, outOfViews), begin_index = attributes
                constructor.addObject(track_id, category)
                for frame_index, bounding_box, occluded, out_of_view in zip(frame_indices, boundingBoxes, occludeds, outOfViews):
                    constructor.addRecord(frame_index, track_id, bounding_box, not(occluded or out_of_view), {'occluded': occluded, 'out_of_view': out_of_view})
            constructor.endInitializingSequence()
