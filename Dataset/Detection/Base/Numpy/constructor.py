from Dataset.Detection.Base.dataset import DetectionDataset
from Dataset.Detection.Base.Numpy.dataset import DetectionDataset_Numpy
import numpy as np
from Dataset.CacheService.constructor import DatasetConstructor_CacheService_Base


class DetectionDatasetConstructor_Numpy(DatasetConstructor_CacheService_Base):
    def __init__(self, dataset: DetectionDataset_Numpy):
        super(DetectionDatasetConstructor_Numpy, self).__init__(dataset)

    def loadFrom(self, dataset: DetectionDataset):
        self.dataset.name = dataset.name
        self.dataset.data_version = dataset.data_version
        self.dataset.data_split = dataset.data_split
        if self.root_path is None:
            self.dataset.root_path = dataset.root_path

        self.dataset.category_names = dataset.category_names
        self.dataset.category_name_id_mapper = dataset.category_name_id_mapper

        image_paths = []
        bounding_boxes = []
        category_ids = []
        image_attributes_indices = []

        current_index = 0
        for image in dataset.images:
            image_attributes_indices.append(current_index)
            image_paths.append(image.image_path)
            for object_ in image.objects:
                bounding_boxes.append(object_.bounding_box)
                category_ids.append(object_.category_id)
                current_index += 1
        self.dataset.image_paths = image_paths
        self.dataset.bounding_boxes = np.array(bounding_boxes)
        self.dataset.category_ids = np.array(category_ids)
        self.dataset.image_attributes_indices = np.array(image_attributes_indices)
