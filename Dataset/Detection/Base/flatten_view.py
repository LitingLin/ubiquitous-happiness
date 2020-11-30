from Dataset.Detection.Base.dataset import DetectionDataset
from Dataset.Detection.Base.object import DetectionDatasetObjectViewer


class DetectionDatasetFlattenView:
    def __init__(self, dataset: DetectionDataset):
        self.dataset = dataset
        self.mapper = []
        for index_of_image, image in enumerate(self.dataset):
            for index_of_object, object_ in enumerate(image):
                self.mapper.append((index_of_image, index_of_object))

    def __len__(self):
        return self.getNumberOfImages()

    def getName(self):
        return self.dataset.name

    def getRootPath(self):
        return str(self.dataset.root_path)

    def getNumberOfImages(self):
        return len(self.mapper)

    def getCategoryName(self, id_: int):
        return self.dataset.category_names[id_]

    def getCategoryId(self, name: str):
        return self.dataset.category_name_id_mapper[name]

    def getNumberOfCategories(self):
        return len(self.dataset.category_names)

    def getCategoryNameList(self):
        return self.dataset.category_names

    def getAttribute(self, name: str):
        return self.dataset.attributes[name]

    def __getitem__(self, index: int):
        return self.getImage(index)

    def getImage(self, index: int):
        mapper = self.mapper[index]
        image = self.dataset.images[mapper[0]]
        object_ = image.objects[mapper[1]]
        return DetectionDatasetObjectViewer(self.dataset, image, object_)

    def hasAttributeCategory(self):
        return self.dataset.hasAttributeCategory()

    def hasAttibuteIsPresent(self):
        return self.dataset.hasAttibuteIsPresent()
