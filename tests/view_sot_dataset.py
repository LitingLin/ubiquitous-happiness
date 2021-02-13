import sys
import os
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_path)

from Dataset.SOT.Seed.UAV123 import UAV123_Seed
from Dataset.Type.data_split import DataSplit
from Dataset.SOT.factory import SingleObjectTrackingDatasetFactory
from Dataset.SOT.Storage.MemoryMapped.Viewer.qt5_viewer import SOTDatasetQt5Viewer

if __name__ == '__main__':
    dataset = SingleObjectTrackingDatasetFactory([UAV123_Seed()]).construct()[0]
    SOTDatasetQt5Viewer(dataset).run()