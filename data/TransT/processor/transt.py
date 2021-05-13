from data.TransT.pipeline import transt_training_preprocessing_pipeline, build_transform
from data.TransT.label.transt import label_generation
import numpy as np


class TransTProcessor:
    def __init__(self,
                 template_size, search_size,
                 template_area_factor, search_area_factor,
                 template_scale_jitter_factor, search_scale_jitter_factor,
                 template_translation_jitter_factor, search_translation_jitter_factor,
                 gray_scale_probability,
                 color_jitter, search_feat_size):
        self.template_size = template_size
        self.search_size = search_size
        self.template_area_factor = template_area_factor
        self.search_area_factor = search_area_factor
        self.template_scale_jitter_factor = template_scale_jitter_factor
        self.search_scale_jitter_factor = search_scale_jitter_factor
        self.template_translation_jitter_factor = template_translation_jitter_factor
        self.search_translation_jitter_factor = search_translation_jitter_factor
        self.gray_scale_probability = gray_scale_probability
        self.search_feat_size = search_feat_size
        self.transform = build_transform(color_jitter)

    def __call__(self, z_image, z_bbox, x_image, x_bbox, _):
        do_gray_scale_transform = np.random.random() < self.gray_scale_probability
        z_image, z_bbox = transt_training_preprocessing_pipeline(z_image, z_bbox, self.template_area_factor,
                                                                 self.template_size,
                                                                 self.template_scale_jitter_factor,
                                                                 self.template_translation_jitter_factor,
                                                                 do_gray_scale_transform,
                                                                 self.transform)
        x_image, x_bbox = transt_training_preprocessing_pipeline(x_image, x_bbox, self.search_area_factor,
                                                                 self.search_size,
                                                                 self.search_scale_jitter_factor,
                                                                 self.search_translation_jitter_factor,
                                                                 do_gray_scale_transform,
                                                                 self.transform)
        target_feat_map_indices, target_class_label_vector, target_bounding_box_label_matrix = label_generation(x_bbox,
                                                                                                                self.search_feat_size,
                                                                                                                self.search_size)
        return z_image, x_image, target_feat_map_indices, target_class_label_vector, target_bounding_box_label_matrix
