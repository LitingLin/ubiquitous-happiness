import numpy as np
from data.operator.bbox.spatial.utility.aligned.normalize_v2 import bbox_normalize
from data.operator.bbox.spatial.xyxy2xywh import bbox_xyxy2xywh
from data.operator.bbox.spatial.xywh2cxcywh import bbox_xywh2cxcywh
import torch
from typing import Optional


def get_target_feat_map_indices(search_feat_size, search_region_size, target_bbox):
    feat_map_indices = np.arange(0, search_feat_size[0] * search_feat_size[1])
    feat_map_indices = feat_map_indices.reshape(search_feat_size[1], search_feat_size[0])

    scale = (
    (search_feat_size[0] - 1) / (search_region_size[0] - 1), (search_feat_size[1] - 1) / (search_region_size[1] - 1))
    target_bbox_feat_map = target_bbox[0] * scale[0], target_bbox[1] * scale[1], target_bbox[2] * scale[0], target_bbox[3] * scale[1]
    pixel_size = (search_feat_size[0] - 1) / search_feat_size[0], (search_feat_size[1] - 1) / search_feat_size[1]
    target_bbox_feat_indices = [v // pixel_size[i % 2] if pixel_size[i % 2] != 0 else 0 for i, v in enumerate(target_bbox_feat_map)]
    # target_bbox_feat_indices = target_bbox_feat_map[0] / pixel_size[0], target_bbox_feat_map[1] / pixel_size[1], target_bbox_feat_map[2] / pixel_size[0], target_bbox_feat_map[3] / pixel_size[1]
    # target_bbox_feat_indices = tuple(int(v) for v in target_bbox_feat_indices)
    target_bbox_feat_indices = feat_map_indices[target_bbox_feat_indices[1]: target_bbox_feat_indices[3] + 1, target_bbox_feat_indices[0]: target_bbox_feat_indices[2] + 1].flatten()

    assert len(target_bbox_feat_indices) != 0
    return torch.tensor(target_bbox_feat_indices, dtype=torch.long)


def generate_target_class_vector(search_feat_size, target_feat_map_indices: Optional[torch.Tensor]):
    target_class_vector = torch.ones([search_feat_size[0] * search_feat_size[1]], dtype=torch.long)
    if target_feat_map_indices is not None:
        target_class_vector[target_feat_map_indices] = 0
    return target_class_vector


def generate_target_bounding_box_label_matrix(bbox, search_region_size, target_feat_map_indices: torch.Tensor):
    length = len(target_feat_map_indices)

    bbox = bbox_xyxy2xywh(bbox)
    bbox = bbox_xywh2cxcywh(bbox)
    bbox = bbox_normalize(bbox, search_region_size)
    bbox = torch.tensor(bbox, dtype=torch.float32)
    return bbox.repeat(length, 1)


def label_generation(bbox, search_feat_size, search_region_size):
    target_feat_map_indices = get_target_feat_map_indices(search_feat_size, search_region_size, bbox)
    target_class_label_vector = generate_target_class_vector(search_feat_size, target_feat_map_indices)
    target_bounding_box_label_matrix = generate_target_bounding_box_label_matrix(bbox, search_region_size, target_feat_map_indices)

    return target_feat_map_indices, target_class_label_vector, target_bounding_box_label_matrix


def negative_label_generation(search_feat_size):
    return None, generate_target_class_vector(search_feat_size, None), None


class TransTLabelGenerator:
    def __init__(self, search_feat_size, search_region_size):
        self.search_feat_size = search_feat_size
        self.search_region_size = search_region_size

    def __call__(self, bbox, is_positive):
        if is_positive:
            return label_generation(bbox, self.search_feat_size, self.search_region_size)
        else:
            return negative_label_generation(self.search_feat_size)
