from .transt import get_target_feat_map_indices
import torch
from typing import Optional
from data.operator.bbox.spatial.normalize import bbox_normalize
from data.types.bounding_box_format import BoundingBoxFormat
from data.operator.bbox.spatial.xyxy2cxcywh import bbox_xyxy2cxcywh


# normalized XYXY and 0 for positive


def generate_target_bounding_box_label_matrix(bbox, search_region_size, target_feat_map_indices: torch.Tensor, target_format : BoundingBoxFormat):
    length = len(target_feat_map_indices)

    bbox = bbox_normalize(bbox, search_region_size)
    if target_format == BoundingBoxFormat.CXCYWH:
        bbox = bbox_xyxy2cxcywh(bbox)

    bbox = torch.tensor(bbox, dtype=torch.float32)
    return bbox.repeat(length, 1)


def generate_target_class_vector(search_feat_size, target_feat_map_indices: Optional[torch.Tensor]):
    target_class_vector = torch.ones([search_feat_size[0] * search_feat_size[1]], dtype=torch.long)
    if target_feat_map_indices is not None:
        target_class_vector[target_feat_map_indices] = 0
    return target_class_vector


def label_generation(bbox, search_feat_size, search_region_size, target_bounding_box_format: BoundingBoxFormat):
    target_feat_map_indices = get_target_feat_map_indices(search_feat_size, search_region_size, bbox)
    target_class_label_vector = generate_target_class_vector(search_feat_size, target_feat_map_indices)
    target_bounding_box_label_matrix = generate_target_bounding_box_label_matrix(bbox, search_region_size, target_feat_map_indices, target_bounding_box_format)

    return target_feat_map_indices, target_class_label_vector, target_bounding_box_label_matrix


def negative_label_generation(search_feat_size):
    return None, generate_target_class_vector(search_feat_size, None), None


class GFocalLabelGenerator:
    def __init__(self, search_feat_size, search_region_size, target_bounding_box_format: BoundingBoxFormat):
        self.search_feat_size = search_feat_size
        self.search_region_size = search_region_size
        self.target_bounding_box_format = target_bounding_box_format
        assert target_bounding_box_format in (BoundingBoxFormat.XYXY, BoundingBoxFormat.CXCYWH)

    def __call__(self, bbox, is_positive):
        if is_positive:
            return label_generation(bbox, self.search_feat_size, self.search_region_size, self.target_bounding_box_format)
        else:
            return negative_label_generation(self.search_feat_size)
