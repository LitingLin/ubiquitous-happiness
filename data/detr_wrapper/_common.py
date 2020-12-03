import torch
from PIL import Image


def _detr_processing(image_path, image_size, image_id, boxes, classes, transforms):
    image = Image.open(image_path).convert('RGB')
    w, h = image_size

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
    if image_id is not None:
        target['image_id'] = torch.as_tensor(image_id)

    if transforms is not None:
        image, target = transforms(image, target)

    return image, target
