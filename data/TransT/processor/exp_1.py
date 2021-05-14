from data.TransT.pipeline import transt_data_processing_train_pipeline, build_transform, transt_data_pre_processing_train_pipeline
from data.TransT.label.exp_1 import label_generation
import numpy as np
from torchvision.transforms import Grayscale


class TransTExp1Processor:
    def __init__(self,
                 template_size, search_size,
                 template_area_factor, search_area_factor,
                 template_scale_jitter_factor, search_scale_jitter_factor,
                 template_translation_jitter_factor, search_translation_jitter_factor,
                 gray_scale_probability,
                 color_jitter, search_feat_size, gaussian_target_label_min_overlap):
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
        self.gaussian_target_label_min_overlap = gaussian_target_label_min_overlap
        self.transform = build_transform(color_jitter)
        self.gray_scale_transformer = Grayscale(3)

    def __call__(self, z_image, z_bbox, x_image, x_bbox, _):
        z_image, x_image = transt_data_pre_processing_train_pipeline(z_image, x_image, self.gray_scale_transformer, self.gray_scale_probability, np.random)
        z_image, z_bbox = transt_data_processing_train_pipeline(z_image, z_bbox, self.template_area_factor,
                                                                 self.template_size,
                                                                 self.template_scale_jitter_factor,
                                                                 self.template_translation_jitter_factor,
                                                                 self.transform)
        x_image, x_bbox = transt_data_processing_train_pipeline(x_image, x_bbox, self.search_area_factor,
                                                                 self.search_size,
                                                                 self.search_scale_jitter_factor,
                                                                 self.search_translation_jitter_factor,
                                                                 self.transform)
        target_feat_map_indices, target_class_label_vector, target_bounding_box_label_matrix = label_generation(x_bbox.tolist(),
                                                                                                                self.search_feat_size,
                                                                                                                self.search_size,
                                                                                                                self.gaussian_target_label_min_overlap)
        return z_image, x_image, target_feat_map_indices, target_class_label_vector, target_bounding_box_label_matrix
