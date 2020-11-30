from Dataset.MOT.Base.constructor import MultipleObjectTrackingDatasetConstructor
import os


def construct_VisDrone2019_MOT(constructor: MultipleObjectTrackingDatasetConstructor, seed):
    sequences_path = seed.root_path
    annotations_path = seed.annotations_path

    sequences = os.listdir(sequences_path)
    sequences.sort()

    for sequence in sequences:
        constructor.beginInitializingSequence()
        constructor.setSequenceName(sequence)
        sequence_path = os.path.join(sequences_path, sequence)
        image_file_names = os.listdir(sequence_path)
        image_file_names.sort()
        for image_file_name in image_file_names:
            image_file_path = os.path.join(sequence_path, image_file_name)
            constructor.addFrame(image_file_path)

        annotation_file_name = sequence + '.txt'
        objects = {}
        object_categories_mapper = {}
        for line in open(os.path.join(annotations_path, annotation_file_name), 'r', encoding='utf-8'):
            line = line.strip()
            if len(line) == 0:
                continue
            words = line.split(',')
            assert len(words) == 10
            '''
                <frame_index>,<target_id>,<bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>
            
             -----------------------------------------------------------------------------------------------------------------------------------
                   Name	                                      Description
             -----------------------------------------------------------------------------------------------------------------------------------
               <frame_index>	  The frame index of the video frame
            
                <target_id>	          In the DETECTION result file, the identity of the target should be set to the constant -1.
                              In the GROUNDTRUTH file, the identity of the target is used to provide the temporal corresponding
                              relation of the bounding boxes in different frames.
            
                <bbox_left>	          The x coordinate of the top-left corner of the predicted bounding box
            
                <bbox_top>	          The y coordinate of the top-left corner of the predicted object bounding box
            
                <bbox_width>	  The width in pixels of the predicted object bounding box
            
                <bbox_height>	  The height in pixels of the predicted object bounding box
            
                  <score>	          The score in the DETECTION file indicates the confidence of the predicted bounding box enclosing
                                      an object instance.
                                      The score in GROUNDTRUTH file is set to 1 or 0. 1 indicates the bounding box is considered in evaluation,
                              while 0 indicates the bounding box will be ignored.
            
              <object_category>	  The object category indicates the type of annotated object, (i.e., ignored regions(0), pedestrian(1),
                                      people(2), bicycle(3), car(4), van(5), truck(6), tricycle(7), awning-tricycle(8), bus(9), motor(10),
                                      others(11))
            
                <truncation>	  The score in the DETECTION file should be set to the constant -1.
                                      The score in the GROUNDTRUTH file indicates the degree of object parts appears outside a frame
                              (i.e., no truncation = 0 (truncation ratio 0%), and partial truncation = 1 (truncation ratio 1% ~ 50%)).
            
                 <occlusion>	  The score in the DETECTION file should be set to the constant -1.
                                      The score in the GROUNDTRUTH file indicates the fraction of objects being occluded
                              (i.e., no occlusion = 0 (occlusion ratio 0%), partial occlusion = 1 (occlusion ratio 1% ~ 50%),
                              and heavy occlusion = 2 (occlusion ratio 50% ~ 100%)).
            '''
            frame_index = int(words[0]) - 1
            target_id = int(words[1])
            bounding_box = [int(words[2]) - 1, int(words[3]) - 1, int(words[4]), int(words[5])]
            object_category = int(words[7])
            truncation = int(words[8])
            occlusion = int(words[9])
            if object_category == 0:
                continue
            if target_id not in objects:
                objects[target_id] = []
            objects[target_id].append((frame_index, bounding_box, truncation, occlusion))
            if target_id not in object_categories_mapper:
                object_categories_mapper[target_id] = []
            object_categories_mapper[target_id].append(object_category)

        for target_id, object_categories in object_categories_mapper.items():
            object_category = max(set(object_categories), key=object_categories.count)
            object_categories_mapper[target_id] = object_category

        for target_id, annotations in objects.items():
            object_category = object_categories_mapper[target_id]
            if object_category == 1:
                object_category = 'pedestrian'
            elif object_category == 2:
                object_category = 'people'
            elif object_category == 3:
                object_category = 'bicycle'
            elif object_category == 4:
                object_category = 'car'
            elif object_category == 5:
                object_category = 'van'
            elif object_category == 6:
                object_category = 'truck'
            elif object_category == 7:
                object_category = 'tricycle'
            elif object_category == 8:
                object_category = 'awning-tricycle'
            elif object_category == 9:
                object_category = 'bus'
            elif object_category == 10:
                object_category = 'motor'
            elif object_category == 11:
                object_category = 'others'
            else:
                raise ValueError
            constructor.addObject(target_id, object_category)
            for frame_index, bounding_box, truncation, occlusion in annotations:
                assert truncation != 2
                constructor.addRecord(frame_index, target_id, bounding_box, occlusion != 2, {'truncation': truncation, 'occlusion': occlusion})
        constructor.endInitializingSequence()
