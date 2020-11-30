from Dataset.Detection.Base.constructor import DetectionDatasetConstructor
import os


def construct_VisDrone2019_DET(constructor: DetectionDatasetConstructor, seed):
    images_path = seed.root_path
    annotation_path = seed.annotation_path
    images = os.listdir(images_path)
    images.sort()
    for image_file_name in images:
        image_name = image_file_name[: -4]
        constructor.beginInitializeImage()
        constructor.setImageName(image_name)
        image_path = os.path.join(images_path, image_file_name)
        constructor.setImagePath(image_path)

        annotation_file_name = image_name + '.txt'
        '''
        <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>


            Name                                                  Description
        -------------------------------------------------------------------------------------------------------------------------------     
         <bbox_left>	     The x coordinate of the top-left corner of the predicted bounding box

         <bbox_top>	     The y coordinate of the top-left corner of the predicted object bounding box

         <bbox_width>	     The width in pixels of the predicted object bounding box

        <bbox_height>	     The height in pixels of the predicted object bounding box

           <score>	     The score in the DETECTION file indicates the confidence of the predicted bounding box enclosing 
                             an object instance.
                             The score in GROUNDTRUTH file is set to 1 or 0. 1 indicates the bounding box is considered in evaluation, 
                             while 0 indicates the bounding box will be ignored.

        <object_category>    The object category indicates the type of annotated object, (i.e., ignored regions(0), pedestrian(1), 
                             people(2), bicycle(3), car(4), van(5), truck(6), tricycle(7), awning-tricycle(8), bus(9), motor(10), 
                             others(11))

        <truncation>	     The score in the DETECTION result file should be set to the constant -1.
                             The score in the GROUNDTRUTH file indicates the degree of object parts appears outside a frame 
                             (i.e., no truncation = 0 (truncation ratio 0%), and partial truncation = 1 (truncation ratio 1% ~ 50%)).

        <occlusion>	     The score in the DETECTION file should be set to the constant -1.
                             The score in the GROUNDTRUTH file indicates the fraction of objects being occluded (i.e., no occlusion = 0 
                             (occlusion ratio 0%), partial occlusion = 1 (occlusion ratio 1% ~ 50%), and heavy occlusion = 2 
                             (occlusion ratio 50% ~ 100%)).
        '''
        for line in open(os.path.join(annotation_path, annotation_file_name), 'r', encoding='utf-8'):
            line = line.strip()
            if len(line) == 0:
                continue
            words = line.split(',')
            words = [word for word in words if len(word) > 0]
            assert len(words) == 8
            object_category = int(words[5])
            if object_category == 0:
                continue
            elif object_category == 1:
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
            truncation = int(words[6])
            occlusion = int(words[7])
            constructor.addObject([int(words[0]) - 1, int(words[1]) - 1, int(words[2]), int(words[3])], object_category,
                                  occlusion != 2,
                                  {'truncation': truncation, 'occlusion': occlusion})
        constructor.endInitializeImage()
