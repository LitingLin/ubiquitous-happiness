from data.tracking.methods.TransT.training.pipeline import TransT_training_data_preprocessing_pipeline, \
    build_TransT_image_augmentation_transformer, TransT_training_image_preprocessing
import numpy as np
from torchvision.transforms import Grayscale


class TransTProcessor:
    def __init__(self,
                 template_size, search_size,
                 template_area_factor, search_area_factor,
                 template_scale_jitter_factor, search_scale_jitter_factor,
                 template_translation_jitter_factor, search_translation_jitter_factor,
                 gray_scale_probability,
                 color_jitter, label_generator, stage2_on_host_process):
        self.template_size = template_size
        self.search_size = search_size
        self.template_area_factor = template_area_factor
        self.search_area_factor = search_area_factor
        self.template_scale_jitter_factor = template_scale_jitter_factor
        self.search_scale_jitter_factor = search_scale_jitter_factor
        self.template_translation_jitter_factor = template_translation_jitter_factor
        self.search_translation_jitter_factor = search_translation_jitter_factor
        self.gray_scale_probability = gray_scale_probability
        if stage2_on_host_process:
            self.transform = None
        else:
            self.transform = build_TransT_image_augmentation_transformer(color_jitter)
        self.gray_scale_transformer = Grayscale(3)
        self.label_generator = label_generator

    def __call__(self, z_image, z_bbox, x_image, x_bbox, is_positive):
        z_image, x_image = TransT_training_image_preprocessing(z_image, x_image, self.gray_scale_transformer,
                                                               self.gray_scale_probability, np.random)
        z_image, z_bbox, z_context = \
            TransT_training_data_preprocessing_pipeline(z_image, z_bbox, self.template_area_factor,
                                                        self.template_size,
                                                        self.template_scale_jitter_factor,
                                                        self.template_translation_jitter_factor,
                                                        self.transform)
        x_image, x_bbox, x_context = \
            TransT_training_data_preprocessing_pipeline(x_image, x_bbox, self.search_area_factor,
                                                        self.search_size,
                                                        self.search_scale_jitter_factor,
                                                        self.search_translation_jitter_factor,
                                                        self.transform)
        target_feat_map_indices, target_class_label_vector, target_bounding_box_label_matrix = \
            self.label_generator(x_bbox.tolist(), is_positive)
        return z_image, x_image, z_context, x_context,\
               target_feat_map_indices, target_class_label_vector, target_bounding_box_label_matrix, is_positive
