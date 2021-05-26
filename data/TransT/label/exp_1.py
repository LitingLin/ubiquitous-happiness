import torch
from data.operator.bbox.spatial.utility.aligned.image import bbox_scale_with_image_resize
from data.operator.bbox.spatial.center import bbox_get_center_point
from data.label.gaussian_target import gaussian_radius, gen_gaussian_target
from .transt import get_target_feat_map_indices, generate_target_bounding_box_label_matrix


def gaussian_target_label(search_feat_size, search_region_size, target_bbox, min_overlap):
    search_feat_width, search_feat_height = search_feat_size
    class_label = torch.zeros((search_feat_height, search_feat_width), dtype=torch.float)

    target_bbox_feat = bbox_scale_with_image_resize(target_bbox, search_region_size, search_feat_size)
    radius = gaussian_radius((search_feat_height, search_feat_width), min_overlap)
    radius = max(round(radius), 1)
    target_bbox_feat_center = bbox_get_center_point(target_bbox_feat)
    target_bbox_feat_center = [round(v) for v in target_bbox_feat_center]
    class_label = gen_gaussian_target(class_label, target_bbox_feat_center, radius)
    return class_label.flatten()


def label_generation(bbox, search_feat_size, search_region_size, gaussian_target_label_min_overlap):
    target_feat_map_indices = get_target_feat_map_indices(search_feat_size, search_region_size, bbox)
    target_class_label_vector = gaussian_target_label(search_feat_size, search_region_size, bbox, gaussian_target_label_min_overlap)
    target_bounding_box_label_matrix = generate_target_bounding_box_label_matrix(bbox, search_region_size, target_feat_map_indices)

    return target_feat_map_indices, target_class_label_vector, target_bounding_box_label_matrix


class Exp1LabelGenerator:
    def __init__(self, search_feat_size, search_region_size, gaussian_target_label_min_overlap):
        self.search_feat_size = search_feat_size
        self.search_region_size = search_region_size
        self.gaussian_target_label_min_overlap = gaussian_target_label_min_overlap

    def __call__(self, bbox):
        return label_generation(bbox, self.search_feat_size, self.search_region_size, self.gaussian_target_label_min_overlap)
