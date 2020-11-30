from Dataset.MOT.Base.constructor import MultipleObjectTrackingDatasetConstructor
import os
import csv
from Dataset.DataSplit import DataSplit


def construct_KITTI_Tracking(constructor: MultipleObjectTrackingDatasetConstructor, seed):
    assert seed.data_split == DataSplit.Training
    images_path = seed.root_path
    labels_path = seed.label_path
    labels_path = os.path.join(labels_path, 'training', 'label_02')

    sequences = os.listdir(images_path)
    sequences.sort()
    for sequence in sequences:
        constructor.beginInitializingSequence()
        constructor.setSequenceName(sequence)
        # KITTI tracking benchmark data format:
        # (frame,tracklet_id,objectType,truncation,occlusion,alpha,x1,y1,x2,y2,h,w,l,X,Y,Z,ry)
        sequence_images_path = os.path.join(images_path, sequence)
        images = os.listdir(sequence_images_path)
        images = [image for image in images if image.endswith('.png')]
        images.sort()

        for image in images:
            constructor.addFrame(os.path.join(sequence_images_path, image))

        tracklet_ids = {}

        annotation_file_path = os.path.join(labels_path, sequence + '.txt')
        with open(annotation_file_path, 'r') as fid:
            reader = csv.reader(fid, delimiter=' ')
            for row in reader:
                object_type = row[2]
                if object_type == 'DontCare':
                    continue
                tracklet_id = int(row[1])
                if tracklet_id not in tracklet_ids:
                    tracklet_ids[tracklet_id] = object_type
                    constructor.addObject(tracklet_id, object_type)
                else:
                    assert tracklet_ids[tracklet_id] == object_type
                frame_index = int(row[0])
                truncation = int(row[3])
                occlusion = int(row[4])
                observation_angle = float(row[5])
                x1 = float(row[6])
                y1 = float(row[7])
                x2 = float(row[8])
                y2 = float(row[9])
                height_m = float(row[10])
                width_m = float(row[11])
                length_m = float(row[12])
                x_m = float(row[13])
                y_m = float(row[14])
                z_m = float(row[15])
                yaw_angle = float(row[16])
                bounding_box = [x1, y1, x2 - x1, y2 - y1]
                is_present = not(truncation == 2 or occlusion == 2)
                constructor.addRecord(frame_index, tracklet_id, bounding_box, is_present)
        constructor.endInitializingSequence()
