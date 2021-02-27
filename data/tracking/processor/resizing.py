import numpy as np
import cv2


def _resize_image_and_bbox_keeping_aspect(image, bbox, target_size):
    h, w = image.shape[0:2]
    scaling_ratio = np.sqrt(target_size / (h * w)).item()
    target_h = int(round(scaling_ratio * h))
    target_w = int(round(scaling_ratio * w))

    target_bbox = np.around(bbox * scaling_ratio)

    return cv2.resize(image, (target_w, target_h)), target_bbox


def _resize_image_keeping_aspect(image, target_size):
    h, w = image.shape[0:2]
    scaling_ratio = np.sqrt(target_size / (h * w)).item()
    target_h = int(round(scaling_ratio * h))
    target_w = int(round(scaling_ratio * w))

    return cv2.resize(image, (target_w, target_h))


class RandomResizing_KeepingAspect_Processor:
    def __init__(self, min_total_size, max_total_size):
        self.min_total_size = min_total_size
        self.max_total_size = max_total_size

    def __call__(self, image: np.ndarray, bbox: np.ndarray):
        target_size = np.random.randint(self.min_total_size, self.max_total_size)
        return _resize_image_and_bbox_keeping_aspect(image, bbox, target_size)


class SizeLimited_KeepingAspect_Processor:
    def __init__(self, size_limitation):
        self.size_limitation = size_limitation

    def __call__(self, image: np.ndarray, bbox: np.ndarray):
        h, w = image.shape[0:2]
        if h * w <= self.size_limitation:
            return image, bbox

        return _resize_image_and_bbox_keeping_aspect(image, bbox, self.size_limitation)


class SizeLimited_KeepingAspect_Image_Processor:
    def __init__(self, size_limitation):
        self.size_limitation = size_limitation

    def __call__(self, image):
        h, w = image.shape[0:2]
        if h * w <= self.size_limitation:
            return image

        return _resize_image_keeping_aspect(image, self.size_limitation)


class SizeLimited_MinMax_KeepingAspect_Image_Processor:
    def __init__(self, min_size_limitation, max_size_limitation):
        self.min_size_limitation = min_size_limitation
        self.max_size_limitation = max_size_limitation

    def __call__(self, image):
        h, w = image.shape[0:2]
        size = h * w
        if size < self.min_size_limitation:
            return _resize_image_keeping_aspect(image, self.min_size_limitation)

        if size > self.max_size_limitation:
            return _resize_image_keeping_aspect(image, self.max_size_limitation)

        return image
