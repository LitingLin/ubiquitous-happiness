import numpy as np
import torch


def generate_mask_from_bbox(img, bbox):
    c, h, w = img.shape
    z_mask = torch.ones((h, w), dtype=torch.bool)
    z_mask[bbox[1]: bbox[1] + bbox[3], bbox[0]: bbox[0] + bbox[2]] = False
    return z_mask


class SiamTransformerMaskGeneratingProcessor:
    def __init__(self, pre_processor):
        self.pre_processor = pre_processor

    def __call__(self, image_z: np.ndarray, z_bounding_box: np.ndarray, image_x: np.ndarray, x_bounding_box: np.ndarray, is_positive):
        z, z_bbox, x, x_bbox = self.pre_processor(image_z, z_bounding_box, image_x, x_bounding_box, is_positive)
        z_mask = generate_mask_from_bbox(z, z_bbox)
        return z, z_mask, x, x_bbox
