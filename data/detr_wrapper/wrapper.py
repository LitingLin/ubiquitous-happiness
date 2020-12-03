from torch.utils.data.dataset import Dataset
from ._common import _detr_processing

class DETRDataset(Dataset):
    def __init__(self, dataset, transforms=None):
        self.dataset = dataset
        self.transforms = transforms

    def __getitem__(self, index: int):
        image_in_dataset = self.dataset[index]
        image_path = image_in_dataset.getImagePath()
        image_size = image_in_dataset.getImageSize()
        image_id = None

        boxes = []
        classes = []
        for object_in_image in image_in_dataset:
            boxes.append(object_in_image.getBoundingBox())
            classes.append(object_in_image.getCategoryId())

        if image_in_dataset.hasAttribute('image_id'):
            image_id = image_in_dataset.getAttribute('image_id')
        return _detr_processing(image_path, image_size, image_id, boxes, classes, self.transforms)

    def __len__(self):
        return len(self.dataset)
