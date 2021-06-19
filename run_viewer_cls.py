from Dataset.CLS.factory import ImageClassificationDatasetFactory
# from Dataset.CLS.Storage.MemoryMapped.Viewer.qt5_viewer import CLSDatasetQt5Viewer
from Dataset.CLS.Seed.ImageNet_21k import ImageNet_21k_Seed


if __name__ == '__main__':
    dataset = ImageClassificationDatasetFactory([ImageNet_21k_Seed()]).construct()[0]
    #CLSDatasetQt5Viewer(dataset).run()
