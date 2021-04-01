from Dataset.SOT.factory import SingleObjectTrackingDatasetFactory
from Dataset.SOT.Seed.TrackingNet import TrackingNet_Seed
import numpy as np
from Miscellaneous.Numpy.dtype import try_get_int_array

if __name__ == '__main__':
    datasets = SingleObjectTrackingDatasetFactory([TrackingNet_Seed()]).construct()

    is_all_int = True
    is_all_float = True
    bounding_boxes_0 = []
    index_begin_from_0 = 0
    bounding_boxes_1 = []
    index_begin_from_1 = 0

    for dataset in datasets:
        for sequence in dataset:
            for frame in sequence:
                if frame.has_bounding_box():
                    if frame.has_bounding_box_validity_flag():
                        if not frame.get_bounding_box_validity_flag():
                            continue
                    bounding_box = frame.get_bounding_box()
                    if bounding_box[0] == 0 or bounding_box[1] == 0:
                        index_begin_from_0 += 1
                        bounding_boxes_0.append(bounding_box)
                    if bounding_box[0] == 1 or bounding_box[1] == 1:
                        index_begin_from_1 += 1
                        bounding_boxes_1.append(bounding_box)
                    if bounding_box.dtype == np.float:
                        if (try_get_int_array(bounding_box).dtype == np.int):
                            is_all_float = False
                        else:
                            is_all_int = False
                    elif bounding_box.dtype == np.int:
                        is_all_float = False
                    else:
                        raise Exception(f'unknown dtype {bounding_box.dtype}')

    print(index_begin_from_0)
    print(index_begin_from_1)

    print(bounding_boxes_0)
    print(bounding_boxes_1)

    print(is_all_int)
    print(is_all_float)
