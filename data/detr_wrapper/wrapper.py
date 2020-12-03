import torch
from torch.utils.data.dataset import Dataset
from PIL import Image


class DETRDataset(Dataset):
    def __init__(self, dataset, transforms=None):
        self.dataset = dataset
        self.transforms = transforms

    def __getitem__(self, index: int):
        image_in_dataset = self.dataset[index]
        image_path = image_in_dataset.getImagePath()
        image = Image.open(image_path).convert('RGB')
        w, h = image_in_dataset.getImageSize()
        boxes = []
        classes = []
        for object_in_image in image_in_dataset:
            boxes.append(object_in_image.getBoundingBox())
            classes.append(object_in_image.getCategoryId())

        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = torch.tensor(classes, dtype=torch.int64)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]

        target = {}
        target['boxes'] = boxes
        target['labels'] = classes
        target['size'] = torch.as_tensor([int(h), int(w)])
        target['orig_size'] = torch.as_tensor([int(h), int(w)])
        if image_in_dataset.hasAttribute('image_id'):
            target['image_id'] = image_in_dataset.getAttribute('image_id')

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self):
        return len(self.dataset)

    def num_classes(self):
        return self.dataset.getNumberOfCategories()
