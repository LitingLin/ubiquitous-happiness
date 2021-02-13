import sys
import os
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_path)

from Dataset.SOT.Seed.OTB100 import OTB100_Seed
from Dataset.Type.data_split import DataSplit
from Dataset.SOT.factory import SingleObjectTrackingDatasetFactory
from evaluation.evaluator.got10k.datasets.otb import OTB

from Dataset.Filter.DataCleaning.Integrity import DataCleaning_Integrity
from Dataset.Filter.DataCleaning.BoundingBox import DataCleaning_BoundingBox

if __name__ == '__main__':
    dataset = SingleObjectTrackingDatasetFactory([OTB100_Seed()]).construct(filters=[DataCleaning_Integrity()], cache_base_format=True, dump_human_readable=True)[0]
    dataset_got = OTB(dataset.get_root_path(), version='tb100')
    assert len(dataset_got) == len(dataset)
    dataset_name_index_map = {}
    for index_of_sequence, sequence in enumerate(dataset):
        dataset_name_index_map[sequence.get_name()] = index_of_sequence
    dataset_name_index_map['Human4'] = dataset_name_index_map['Human4_2']
    dataset_name_index_map['Jogging.1'] = dataset_name_index_map['Jogging_1']
    dataset_name_index_map['Jogging.2'] = dataset_name_index_map['Jogging_2']
    dataset_name_index_map['Skating2.1'] = dataset_name_index_map['Skating2_1']
    dataset_name_index_map['Skating2.2'] = dataset_name_index_map['Skating2_2']
    for index in range(len(dataset)):
        sequence_name = dataset_got.seq_names[index]
        if sequence_name == 'Tiger1':
            continue
        sequence = dataset[dataset_name_index_map[sequence_name]]
        _, annos = dataset_got[index]
        bounding_boxes = sequence.get_all_bounding_box()
        assert len(sequence) == annos.shape[0]
        valid_annotation_indices = sequence.get_all_bounding_box_validity_flag()
        if valid_annotation_indices is not None:
            bounding_boxes = bounding_boxes[valid_annotation_indices]
            annos = annos[valid_annotation_indices]

        assert (bounding_boxes == annos).all()
