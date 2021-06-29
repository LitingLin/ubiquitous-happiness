from Dataset.CLS.factory import ImageClassificationDatasetFactory
from Dataset.CLS.Storage.MemoryMapped.Viewer.qt5_viewer import CLSDatasetQt5Viewer
from Dataset.CLS.Seed.ImageNet_21k import ImageNet_21k_Seed
from Dataset.Filter.DataCleaning.Integrity import DataCleaning_Integrity


if __name__ == '__main__':
    dataset = ImageClassificationDatasetFactory([ImageNet_21k_Seed()]).construct(filters=[DataCleaning_Integrity()])[0]
    CLSDatasetQt5Viewer(dataset).run()
