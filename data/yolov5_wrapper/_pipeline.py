from Dataset.Detection.Base.MemoryMapped.dataset import DetectionDataset_MemoryMapped
import cv2
from data.operator.bbox.yolov5.letterbox import letterbox
import numpy as np
from data.augmentation.yolov5.random_perspective import random_perspective
from data.augmentation.yolov5.augment_hsv import augment_hsv
from data.operator.bbox.yolov5.xyxy2xywh import xyxy2xywh
import random
import torch


def image_loading_pipeline(dataset: DetectionDataset_MemoryMapped, index: int, target_image_size, do_augmentation, config: dict, rectangular_output:bool, batch_size, stride, pad):
    # loads 1 image from dataset, returns img, original hw, resized hw
    image = dataset[index]
    img = cv2.imread(image.getImagePath())

    h0, w0 = img.shape[:2]  # orig hw
    r = target_image_size / max(h0, w0)  # resize image to img_size

    if r != 1:  # always resize down, only resize up if training with augmentation
        interp = cv2.INTER_AREA if r < 1 and not do_augmentation else cv2.INTER_LINEAR
        img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)

    h, w = img.shape[:2]

    # Letterbox

    if rectangular_output:
        # Set training image shapes
        bi = index // batch_size
        sizes = dataset.getAllImageSize()[bi * batch_size: (bi + 1) * batch_size]
        ar = sizes[:, 1] / sizes[:, 0]  # aspect ratio
        mini, maxi = ar.min(), ar.max()
        shape = [1, 1]
        if maxi < 1:
            shape = [maxi, 1]
        elif mini > 1:
            shape = [1, 1 / mini]

        shape = np.ceil(np.array(shape) * target_image_size / stride + pad).astype(np.int) * stride
    else:
        shape = target_image_size

    img, ratio, pad = letterbox(img, shape, auto=False, scaleup=do_augmentation)
    shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

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

    labels[:, 1] = labels[:, 1] * ratio[0] + pad[0]
    labels[:, 2] = labels[:, 2] * ratio[1] + pad[1]
    labels[:, 3] = labels[:, 3] * ratio[0] + pad[0]
    labels[:, 4] = labels[:, 4] * ratio[1] + pad[1]

    if do_augmentation:
        # Augment imagespace
        img, labels = random_perspective(img, labels,
                                         degrees=config['degrees'],
                                         translate=config['translate'],
                                         scale=config['scale'],
                                         shear=config['shear'],
                                         perspective=config['perspective'])

        # Augment colorspace
        augment_hsv(img, hgain=config['hsv_h'], sgain=config['hsv_s'], vgain=config['hsv_v'])

    nL = labels.shape[0]  # number of labels
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

    return torch.from_numpy(img), labels_out, image.getImagePath(), shapes
