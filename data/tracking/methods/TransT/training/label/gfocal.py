from .transt import get_target_feat_map_indices
import torch
from typing import Optional
from data.operator.bbox.spatial.normalize import bbox_normalize


# normalized XYXY and 0 for positive


def generate_target_bounding_box_label_matrix(bbox, search_region_size, target_feat_map_indices: torch.Tensor):
    length = len(target_feat_map_indices)

    bbox = bbox_normalize(bbox, search_region_size)
    bbox = torch.tensor(bbox, dtype=torch.float32)
    return bbox.repeat(length, 1)


def generate_target_class_vector(search_feat_size, target_feat_map_indices: Optional[torch.Tensor]):
    target_class_vector = torch.ones([search_feat_size[0] * search_feat_size[1]], dtype=torch.long)
    if target_feat_map_indices is not None:
        target_class_vector[target_feat_map_indices] = 0
    return target_class_vector


def label_generation(bbox, search_feat_size, search_region_size):
    target_feat_map_indices = get_target_feat_map_indices(search_feat_size, search_region_size, bbox)
    target_class_label_vector = generate_target_class_vector(search_feat_size, target_feat_map_indices)
    target_bounding_box_label_matrix = generate_target_bounding_box_label_matrix(bbox, search_region_size, target_feat_map_indices)

    return target_feat_map_indices, target_class_label_vector, target_bounding_box_label_matrix


def negative_label_generation(search_feat_size):
    return None, generate_target_class_vector(search_feat_size, None), None


class GFocalLabelGenerator:
    def __init__(self, search_feat_size, search_region_size):
        self.search_feat_size = search_feat_size
        self.search_region_size = search_region_size

    def __call__(self, bbox, is_positive):
        if is_positive:
            return label_generation(bbox, self.search_feat_size, self.search_region_size)
        else:
            return negative_label_generation(self.search_feat_size)
