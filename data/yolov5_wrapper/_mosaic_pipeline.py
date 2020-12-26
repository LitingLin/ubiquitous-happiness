from Dataset.Detection.Base.MemoryMapped.dataset import DetectionDataset_MemoryMapped
import random
import cv2
import numpy as np
from data.augmentation.yolov5.random_perspective import random_perspective
from data.augmentation.yolov5.augment_hsv import augment_hsv
from data.operator.bbox.yolov5.xyxy2xywh import xyxy2xywh
import torch


def _load_image_as_mosaic(dataset: DetectionDataset_MemoryMapped, index: int, target_image_size: int, do_augmentation: bool, config: dict):
    mosaic_border = [-target_image_size // 2, -target_image_size // 2]
    # loads images in a mosaic

    labels4 = []
    s = target_image_size
    yc, xc = [int(random.uniform(-x, 2 * s + x)) for x in mosaic_border]  # mosaic center x, y
    indices = [index] + [random.randint(0, len(dataset) - 1) for _ in range(3)]  # 3 additional image indices
    for i, index in enumerate(indices):
        image = dataset[index]
        img = cv2.imread(image.getImagePath())
        h0, w0 = img.shape[:2]  # orig hw
        r = target_image_size / max(h0, w0)  # resize image to img_size
        if r != 1:  # always resize down, only resize up if training with augmentation
            interp = cv2.INTER_AREA if r < 1 and not do_augmentation else cv2.INTER_LINEAR
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
        h, w = img.shape[:2]

        # place img in img4
        if i == 0:  # top left
            img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        padw = x1a - x1b
        padh = y1a - y1b

        bboxes = image.getAllBoundingBox()
        assert image.hasAttributeCategory()
        category_ids = image.getAllCategoryId()
        if image.hasAttributeIsPresent():
            bboxes = bboxes[image.getAllAttributeIsPresent()]
            category_ids = category_ids[image.getAllAttributeIsPresent()]

        labels = np.ascontiguousarray(np.concatenate((category_ids[:, None], bboxes), axis=1))

        # To xyxy format
        labels[:, 1] = labels[:, 1] / w0 * w
        labels[:, 2] = labels[:, 2] / h0 * h
        labels[:, 3] = labels[:, 3] / w0 * w
        labels[:, 4] = labels[:, 4] / h0 * h

        labels[:, 3] = labels[:, 1] + labels[:, 3]
        labels[:, 4] = labels[:, 2] + labels[:, 4]

        labels[:, 1] = labels[:, 1] + padw
        labels[:, 2] = labels[:, 2] + padh
        labels[:, 3] = labels[:, 3] + padw
        labels[:, 4] = labels[:, 4] + padh

        labels4.append(labels)

    # Concat/clip labels
    if len(labels4):
        labels4 = np.concatenate(labels4, 0)
        np.clip(labels4[:, 1:], 0, 2 * s, out=labels4[:, 1:])  # use with random_perspective
        # img4, labels4 = replicate(img4, labels4)  # replicate

    # Augment
    img4, labels4 = random_perspective(img4, labels4,
                                       degrees=config['degrees'],
                                       translate=config['translate'],
                                       scale=config['scale'],
                                       shear=config['shear'],
                                       perspective=config['perspective'],
                                       border=mosaic_border)  # border to remove

    return img4, labels4

def mosaic_image_loading_pipeline(dataset: DetectionDataset_MemoryMapped, index: int, target_image_size: int, do_augmentation: bool, config: dict):
    # Load mosaic
    img, labels = _load_image_as_mosaic(dataset, index, target_image_size, do_augmentation, config)
    shapes = None

    # MixUp https://arxiv.org/pdf/1710.09412.pdf
    if random.random() < config['mixup']:
        img2, labels2 = _load_image_as_mosaic(dataset, random.randint(0, len(dataset) - 1), target_image_size, do_augmentation, config)
        r = np.random.beta(8.0, 8.0)  # mixup ratio, alpha=beta=8.0
        img = (img * r + img2 * (1 - r)).astype(np.uint8)
        labels = np.concatenate((labels, labels2), 0)

    if do_augmentation:
        # Augment colorspace
        augment_hsv(img, hgain=config['hsv_h'], sgain=config['hsv_s'], vgain=config['hsv_v'])

        # Apply cutouts
        # if random.random() < 0.9:
        #     labels = cutout(img, labels)

    nL = len(labels)  # number of labels
    if nL:
        labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])  # convert xyxy to xywh
        labels[:, [2, 4]] /= img.shape[0]  # normalized height 0-1
        labels[:, [1, 3]] /= img.shape[1]  # normalized width 0-1

    if do_augmentation:
        # flip up-down
        if random.random() < config['flipud']:
            img = np.flipud(img)
            if nL:
                labels[:, 2] = 1 - labels[:, 2]

        # flip left-right
        if random.random() < config['fliplr']:
            img = np.fliplr(img)
            if nL:
                labels[:, 1] = 1 - labels[:, 1]

    labels_out = torch.zeros((nL, 6))
    if nL:
        labels_out[:, 1:] = torch.from_numpy(labels)

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    return torch.from_numpy(img), labels_out, dataset[index].getImagePath(), shapes
