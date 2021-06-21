from Dataset.CLS.factory import ImageClassificationDatasetFactory
# from Dataset.CLS.Storage.MemoryMapped.Viewer.qt5_viewer import CLSDatasetQt5Viewer
from Dataset.CLS.Seed.ImageNet_21k import ImageNet_21k_Seed
from training.SOT.datasets import get_standard_trainning_dataset_filter


if __name__ == '__main__':
    dataset = ImageClassificationDatasetFactory([ImageNet_21k_Seed()]).construct(filters=get_standard_trainning_dataset_filter())[0]
    #CLSDatasetQt5Viewer(dataset).run()
