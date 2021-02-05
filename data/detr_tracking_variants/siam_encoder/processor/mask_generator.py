import numpy as np
import torch


class SiamTransformerMaskGeneratingProcessor:
    def __init__(self, pre_processor):
        self.pre_processor = pre_processor

    def __call__(self, image_z: np.ndarray, z_bounding_box: np.ndarray, image_x: np.ndarray, x_bounding_box: np.ndarray, is_positive):
        z, z_bbox, x, x_bbox = self.pre_processor(image_z, z_bounding_box, image_x, x_bounding_box, is_positive)

        c, h, w = z.shape
        z_mask = torch.ones((h, w), dtype=torch.bool)
        z_mask[z_bbox[1]: z_bbox[1] + z_bbox[3], z_bbox[0]: z_bbox[0] + z_bbox[2]] = False
        return z, z_mask, x, x_bbox
