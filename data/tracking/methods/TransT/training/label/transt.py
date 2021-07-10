import numpy as np
from data.operator.bbox.spatial.xyxy2xywh import bbox_xyxy2xywh
from data.operator.bbox.spatial.xywh2cxcywh import bbox_xywh2cxcywh
from data.types.bounding_box_format import BoundingBoxFormat
import torch
from typing import Optional


def get_target_feat_map_indices(search_feat_size, search_region_size, target_bbox):
    feat_map_indices = np.arange(0, search_feat_size[0] * search_feat_size[1])
    feat_map_indices = feat_map_indices.reshape(search_feat_size[1], search_feat_size[0])

    scale = (
    (search_feat_size[0] - 1) / (search_region_size[0] - 1), (search_feat_size[1] - 1) / (search_region_size[1] - 1))
    target_bbox_feat_map = target_bbox[0] * scale[0], target_bbox[1] * scale[1], target_bbox[2] * scale[0], target_bbox[3] * scale[1]
    pixel_size = (search_feat_size[0] - 1) / search_feat_size[0], (search_feat_size[1] - 1) / search_feat_size[1]
    target_bbox_feat_indices = [int(v // pixel_size[i % 2]) if pixel_size[i % 2] != 0 else 0 for i, v in enumerate(target_bbox_feat_map)]
    # target_bbox_feat_indices = target_bbox_feat_map[0] / pixel_size[0], target_bbox_feat_map[1] / pixel_size[1], target_bbox_feat_map[2] / pixel_size[0], target_bbox_feat_map[3] / pixel_size[1]
    # target_bbox_feat_indices = tuple(int(v) for v in target_bbox_feat_indices)
    target_bbox_feat_indices = feat_map_indices[target_bbox_feat_indices[1]: target_bbox_feat_indices[3] + 1, target_bbox_feat_indices[0]: target_bbox_feat_indices[2] + 1].flatten()

    assert len(target_bbox_feat_indices) != 0
    return torch.tensor(target_bbox_feat_indices, dtype=torch.long)


def _get_featmap_element_coord_matrix(search_feat_size, search_region_size):
    x = torch.linspace(0, search_region_size[0] - 1, search_feat_size[0] * 2 + 1)
    x = x[1::2]
    y = torch.linspace(0, search_region_size[1] - 1, search_feat_size[1] * 2 + 1)
    y = y[1::2]
    return x, y


def get_target_feat_map_indices_centerness(search_feat_size, search_region_size, target_bbox):
    x_feat_center, y_feat_center = _get_featmap_element_coord_matrix(search_feat_size, search_region_size)
    x_indices = x_feat_center >= target_bbox[0] & x_feat_center <= target_bbox[2]
    y_indices = y_feat_center >= target_bbox[1] & y_feat_center <= target_bbox[3]
    if len(x_indices) == 0:
        x_center = (target_bbox[0] + target_bbox[2]) / 2
        x_diff = torch.abs(x_feat_center - x_center)
        x_indices = torch.min(x_diff) == x_diff
    if len(y_indices) == 0:
        y_center = (target_bbox[1] + target_bbox[3]) / 2
        y_diff = torch.abs(y_feat_center - y_center)
        y_indices = torch.min(y_diff) == y_diff

    feat_map_indices = torch.arange(0, search_feat_size[0] * search_feat_size[1], dtype=torch.long)
    feat_map_indices = feat_map_indices.reshape(search_feat_size[1], search_feat_size[0])
    return feat_map_indices[y_indices, x_indices]


def get_target_feat_map_indices_single(search_feat_size, search_region_size, target_bbox):
    return torch.tensor([0], dtype=torch.long)


def generate_target_class_vector(search_feat_size, target_feat_map_indices: Optional[torch.Tensor]):
    target_class_vector = torch.ones([search_feat_size[0] * search_feat_size[1]], dtype=torch.long)
    if target_feat_map_indices is not None:
        target_class_vector[target_feat_map_indices] = 0
    return target_class_vector


def generate_target_bounding_box_label_matrix(bbox, search_region_size, target_feat_map_indices: torch.Tensor, bounding_box_format: BoundingBoxFormat, bbox_normalizer):
    length = len(target_feat_map_indices)

    if bounding_box_format == BoundingBoxFormat.CXCYWH:
        bbox = bbox_xyxy2xywh(bbox)
        bbox = bbox_xywh2cxcywh(bbox)
    bbox = bbox_normalizer.normalize(bbox, search_region_size)
    bbox = torch.tensor(bbox, dtype=torch.float32)
    return bbox.repeat(length, 1)


def label_generation(bbox, search_feat_size, search_region_size, positive_sample_assignment_fn, bounding_box_format, bbox_normalizer):
    target_feat_map_indices = positive_sample_assignment_fn(search_feat_size, search_region_size, bbox)
    target_class_label_vector = generate_target_class_vector(search_feat_size, target_feat_map_indices)
    target_bounding_box_label_matrix = generate_target_bounding_box_label_matrix(bbox, search_region_size, target_feat_map_indices, bounding_box_format, bbox_normalizer)

    return target_feat_map_indices, target_class_label_vector, target_bounding_box_label_matrix


def negative_label_generation(search_feat_size):
    return None, generate_target_class_vector(search_feat_size, None), None


class TransTLabelGenerator:
    def __init__(self, search_feat_size, search_region_size, positive_label_assignment_method,
                 target_bounding_box_format,
                 bounding_box_normalization_helper):
        self.search_feat_size = search_feat_size
        self.search_region_size = search_region_size
        self.target_bounding_box_format = target_bounding_box_format
        assert target_bounding_box_format in (BoundingBoxFormat.XYXY, BoundingBoxFormat.CXCYWH)
        self.positive_label_assignment_method = positive_label_assignment_method
        assert positive_label_assignment_method in ('round', 'centerness')
        if search_feat_size[0] * search_region_size[0] == 1:
            self.positive_label_assignment_method = get_target_feat_map_indices_single
        elif positive_label_assignment_method == 'round':
            self.positive_sample_assignment_fn = get_target_feat_map_indices
        elif positive_label_assignment_method == 'centerness':
            self.positive_label_assignment_method = get_target_feat_map_indices_centerness
        else:
            raise NotImplementedError
        self.bounding_box_normalization_helper = bounding_box_normalization_helper

    def __call__(self, bbox, is_positive):
        if is_positive:
            return label_generation(bbox, self.search_feat_size, self.search_region_size, self.positive_sample_assignment_fn, self.target_bounding_box_format, self.bounding_box_normalization_helper)
        else:
            return negative_label_generation(self.search_feat_size)
