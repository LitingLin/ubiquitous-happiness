from Dataset.SOT.factory import SingleObjectTrackingDatasetFactory
from Dataset.SOT.Seed.LaSOT import LaSOT_Seed

if __name__ == '__main__':
    datasets = SingleObjectTrackingDatasetFactory([LaSOT_Seed()]).construct()
    index_begin_from_0 = 0
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
                    if bounding_box[0] == 1 or bounding_box[1] == 1:
                        index_begin_from_1 += 1

    print(index_begin_from_0)
    print(index_begin_from_1)
