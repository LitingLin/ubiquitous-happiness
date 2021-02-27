import sys
import os
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_path)

from Dataset.MOT.Seed.YoutubeBB import YoutubeBB_Seed
from Dataset.Type.data_split import DataSplit
from Dataset.MOT.factory import MultipleObjectTrackingDatasetFactory
from Dataset.MOT.Storage.MemoryMapped.Viewer.qt5_viewer import MOTDatasetQt5Viewer
from Dataset.Filter.DataCleaning.Integrity import DataCleaning_Integrity
from Dataset.Filter.DataCleaning.BoundingBox import DataCleaning_BoundingBox

if __name__ == '__main__':
    dataset = MultipleObjectTrackingDatasetFactory([YoutubeBB_Seed(data_split=DataSplit.Training)]).construct([DataCleaning_BoundingBox(update_validity=True, remove_non_validity_objects=True, remove_empty_annotation_objects=True), DataCleaning_Integrity(remove_zero_annotation_image=True, remove_zero_annotation_video_head_tail=True, remove_invalid_image=True)])[0]
    MOTDatasetQt5Viewer(dataset).run()