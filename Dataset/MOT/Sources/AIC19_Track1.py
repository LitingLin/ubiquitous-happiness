from Dataset.MOT.FactorySeeds.AIC19_Track1 import AIC19_Track1_Seed
from Dataset.MOT.Base.constructor import MultipleObjectTrackingDatasetConstructor
from Dataset.Utils.decode_video_to_path import decode_video_file
import os
import shutil
from Dataset.DataSplit import DataSplit


def construct_AIC19Track1(constructor: MultipleObjectTrackingDatasetConstructor, seed: AIC19_Track1_Seed):
    assert seed.data_split == DataSplit.Training
    scenes_path = os.path.join(seed.root_path, 'train')
    scenes = os.listdir(scenes_path)
    scenes.sort()
    for scene in scenes:
        scene_path = os.path.join(scenes_path, scene)
        cameras = os.listdir(scene_path)
        cameras.sort()
        for camera in cameras:
            camera_path = os.path.join(scene_path, camera)
            images_path = os.path.join(camera_path, 'imgs')
            if not os.path.exists(images_path):
                tmp_path = os.path.join(camera_path, 'tmp')
                if os.path.exists(tmp_path):
                    shutil.rmtree(tmp_path)
                os.mkdir(tmp_path)
                decode_video_file(os.path.join(camera_path, 'vdo.avi'), tmp_path)
                os.rename(tmp_path, images_path)
            images = os.listdir(images_path)
            images.sort()
            constructor.beginInitializingSequence()
            constructor.setSequenceName('{}-{}'.format(scene, camera))
            for image in images:
                constructor.addFrame(os.path.join(images_path, image))
            gt_file_path = os.path.join(camera_path, 'gt', 'gt.txt')
            ids = {}
            # [frame, ID, left, top, width, height, 1, -1, -1, -1]
            for line in open(gt_file_path):
                line = line.strip()
                if len(line) == 0:
                    continue
                values = line.split(',')
                assert len(values) == 10
                frame_index = int(values[0]) - 1
                object_id = int(values[1])
                if object_id not in ids:
                    ids[object_id] = None
                    constructor.addObject(object_id, 'vehicle')
                constructor.addRecord(frame_index, object_id,
                                      [int(values[2]), int(values[3]), int(values[4]), int(values[5])])
            constructor.endInitializingSequence()
