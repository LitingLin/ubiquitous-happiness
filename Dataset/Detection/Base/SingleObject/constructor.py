from Dataset.Detection.Base.SingleObject.dataset import SingleObjectDetectionDataset
from Dataset.Detection.Base.SingleObject.image import SingleObjectDetectionDatasetImage
from typing import List, Dict
import pathlib



class SingleObjectDetectionConstructor:
    def __init__(self, dataset: SingleObjectDetectionDataset):
        self.dataset = dataset

    def setName(self, name: str):
        self.dataset.name = name

    def setRootPath(self, path: str):
        path = pathlib.Path(path)
        assert path.exists()

        self.dataset.root_path = path

    def addImage(self, path: str, name: str, bounding_box: List, category_name: str, additional_attributes: Dict=None):
        image = SingleObjectDetectionDatasetImage()
        image.name = name
        path = pathlib.Path(path)
        assert path.exists()
        image.image_path = path.relative_to(self.dataset.root_path)

        image.bounding_box = bounding_box
        if category_name not in self.dataset.category_name_id_mapper:
            id_ = len(self.dataset.category_names)
            self.dataset.category_names.append(category_name)
            self.dataset.category_name_id_mapper[category_name] = id_
        else:
            id_ = self.dataset.category_name_id_mapper[category_name]
        image.category_id = id_
        image.attributes = additional_attributes

        self.dataset.images.append(image)

