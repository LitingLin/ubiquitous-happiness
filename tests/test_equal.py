import sys
import os
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_path)

from Dataset.SOT.Seed.UAV123 import UAV123_Seed
from Dataset.Type.data_split import DataSplit
from Dataset.SOT.factory import SingleObjectTrackingDatasetFactory
from evaluation.evaluator.got10k.datasets.uav123 import UAV123

from Dataset.Filter.DataCleaning.Integrity import DataCleaning_Integrity
from Dataset.Filter.DataCleaning.BoundingBox import DataCleaning_BoundingBox

if __name__ == '__main__':
    dataset = SingleObjectTrackingDatasetFactory([UAV123_Seed()]).construct(filters=[DataCleaning_BoundingBox(), DataCleaning_Integrity()])[0]
    dataset_got = UAV123(dataset.get_root_path())
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
