import sys
import os
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_path)

from Dataset.SOT.Seed.NFS import NFS_Seed, NFSDatasetVersionFlag
from Dataset.Type.data_split import DataSplit
from Dataset.SOT.factory import SingleObjectTrackingDatasetFactory
from evaluation.evaluator.got10k.datasets.nfs import NfS

from Dataset.Filter.DataCleaning.Integrity import DataCleaning_Integrity
from Dataset.Filter.DataCleaning.BoundingBox import DataCleaning_BoundingBox

if __name__ == '__main__':
    dataset = SingleObjectTrackingDatasetFactory([NFS_Seed(version=NFSDatasetVersionFlag.fps_240)]).construct(filters=None)[0]
    dataset_got = NfS(dataset.get_root_path(), 240)
    assert len(dataset_got) == len(dataset)
    dataset_name_index_map = {}
    for index_of_sequence, sequence in enumerate(dataset):
        dataset_name_index_map[sequence.get_name()] = index_of_sequence
    for index in range(len(dataset)):
        sequence_name = dataset_got.seq_names[index]
        sequence = dataset[dataset_name_index_map[sequence_name]]
        _, annos = dataset_got[index]
        assert len(sequence) == annos.shape[0]
        bounding_boxes = sequence.get_all_bounding_box()
        valid_annotation_indices = sequence.get_all_bounding_box_validity_flag()
        if valid_annotation_indices is not None:
            bounding_boxes = bounding_boxes[valid_annotation_indices]
            annos = annos[valid_annotation_indices]

        assert (bounding_boxes == annos).all()
