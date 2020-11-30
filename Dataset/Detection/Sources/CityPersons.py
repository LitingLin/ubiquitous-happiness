import scipy.io
from Dataset.MetaInformation.cityscapes_person import id2labelCp
import numpy
import os
from typing import Dict
from Dataset.DataSplit import DataSplit


# rich person class labels for cityscapes
def construct_cityPersons(constructor, seed):
    data_split = seed.data_split
    cityscapes_path = seed.root_path
    annotation_path = seed.annotation_path
    things_only = seed.things_only

    def _parse_cityPersons(path: str, mat: Dict):
        if 'anno_train_aligned' in mat:
            annotations: numpy.ndarray = mat['anno_train_aligned']
        elif 'anno_val_aligned' in mat:
            annotations: numpy.ndarray = mat['anno_val_aligned']
        else:
            raise Exception
        anno_shape = annotations.shape
        assert len(anno_shape) == 2
        assert anno_shape[0] == 1
        number_of_annotations = anno_shape[1]
        for index_of_annotations in range(number_of_annotations):
            annotation = annotations[0][index_of_annotations][0][0]
            city_name = str(annotation['cityname'][0])
            im_name = str(annotation['im_name'][0])
            bbs = annotation['bbs']
            bbs_shape = bbs.shape
            assert len(bbs_shape) == 2
            # [class_label, x1,y1,w,h, instance_id, x1_vis, y1_vis, w_vis, h_vis]
            assert bbs_shape[1] == 10
            '''            
    class_label =0: ignore regions (fake humans, e.g. people on posters, reflections etc.)
    class_label =1: pedestrians
    class_label =2: riders
    class_label =3: sitting persons
    class_label =4: other persons with unusual postures
    class_label =5: group of people
            '''
            constructor.beginInitializeImage()
            constructor.setImageName(im_name)
            constructor.setImagePath(os.path.join(path, city_name, im_name))
            for index_of_object in range(bbs_shape[0]):
                class_label = int(bbs[index_of_object][0])
                class_label = id2labelCp[class_label]
                if things_only:
                    if not class_label.hasInstances:
                        continue
                bounding_box = bbs[index_of_object][6:10].tolist()
                category_name = class_label.name
                constructor.addObject(bounding_box, category_name)
            constructor.endInitializeImage()

    if data_split & DataSplit.Training:
        _parse_cityPersons(os.path.join(cityscapes_path, 'train'), scipy.io.loadmat(os.path.join(annotation_path, 'anno_train.mat')))
    if data_split & DataSplit.Validation:
        _parse_cityPersons(os.path.join(cityscapes_path, 'val'), scipy.io.loadmat(os.path.join(annotation_path, 'anno_val.mat')))
