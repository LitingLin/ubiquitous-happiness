import sys
import os
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_path)

from Dataset.SOT.Seed.LaSOT import LaSOT_Seed
from Dataset.Type.data_split import DataSplit
from Dataset.SOT.factory import SingleObjectTrackingDatasetFactory
from evaluation.evaluator.got10k.datasets.lasot import LaSOT


if __name__ == '__main__':
    dataset = SingleObjectTrackingDatasetFactory([LaSOT_Seed(data_split=DataSplit.Validation)]).construct()[0]
    dataset_got = LaSOT(dataset.get_root_path())
    assert len(dataset_got) == len(dataset)
    dataset_name_index_map = {}
    for index_of_sequence, sequence in enumerate(dataset):
        dataset_name_index_map[sequence.get_name()] = index_of_sequence
    for index in range(len(dataset)):
        sequence_name = dataset_got.seq_names[index]
        sequence = dataset[dataset_name_index_map[sequence_name]]
        _, annos = dataset_got[index]
        assert len(sequence) == annos.shape[0]
        assert (sequence.get_all_bounding_box() == annos).all()
