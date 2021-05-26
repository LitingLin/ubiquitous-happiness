import numpy as np
from data.operator.bbox.transform.rasterize.aligned import bbox_rasterize_aligned
from data.operator.bbox.spatial.utility.aligned.image import bbox_scale_with_image_resize
from data.operator.bbox.spatial.utility.aligned.normalize_v2 import bbox_normalize
from data.operator.bbox.spatial.xyxy2xywh import bbox_xyxy2xywh
from data.operator.bbox.spatial.xywh2cxcywh import bbox_xywh2cxcywh
import torch
from typing import Optional


def get_target_feat_map_indices(search_feat_size, search_region_size, target_bbox):
    feat_map_indices = np.arange(0, search_feat_size[0] * search_feat_size[1])

    feat_map_indices = feat_map_indices.reshape(search_feat_size)

    target_bbox_ = bbox_scale_with_image_resize(target_bbox, search_region_size, search_feat_size)
    target_bbox_ = bbox_rasterize_aligned(target_bbox_)

    target_feat_map_indices = feat_map_indices[target_bbox_[1]: target_bbox_[3] + 1, target_bbox_[0]: target_bbox_[2] + 1].flatten()
    assert len(target_feat_map_indices) != 0
    return torch.tensor(target_feat_map_indices, dtype=torch.long)


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


def get_bounding_box_from_label(label, search_region_size):
    label = label.tolist()

    from data.operator.bbox.spatial.utility.aligned.normalize_v2 import bbox_denormalize
    from data.operator.bbox.spatial.xywh2xyxy import bbox_xywh2xyxy
    from data.operator.bbox.spatial.cxcywh2xywh import bbox_cxcywh2xywh

    bbox = bbox_denormalize(label, search_region_size)
    bbox = bbox_cxcywh2xywh(bbox)
    return bbox_xywh2xyxy(bbox)


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
