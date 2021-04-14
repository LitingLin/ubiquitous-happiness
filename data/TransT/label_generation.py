import numpy as np
from data.operator.bbox.transform.rasterize.aligned import bbox_rasterize_aligned
from data.operator.bbox.spatial.utility.aligned.image import bbox_scale_with_image_resize
from data.operator.bbox.spatial.utility.aligned.normalize_v2 import bbox_normalize
from data.operator.bbox.spatial.xyxy2xywh import bbox_xyxy2xywh
from data.operator.bbox.spatial.xywh2cxcywh import bbox_xywh2cxcywh
import torch


def get_target_feat_map_indices(search_feat_size, search_region_size, target_bbox):
    feat_map_indices = np.arange(0, search_feat_size[0] * search_feat_size[1])

    feat_map_indices.reshape(search_feat_size)

    target_bbox = np.array(target_bbox, dtype=np.float)
    target_bbox = bbox_scale_with_image_resize(target_bbox, search_region_size, search_feat_size)
    target_bbox = bbox_rasterize_aligned(target_bbox)
    target_feat_map_indices = feat_map_indices[target_bbox[1]: target_bbox[3] + 1, target_bbox[0]: target_bbox[2] + 1]
    return torch.tensor(target_feat_map_indices)


def generate_target_class_vector(search_feat_size, target_feat_map_indices: torch.Tensor):
    target_class_vector = torch.ones([search_feat_size[0] * search_feat_size[1]], dtype=torch.long)
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
