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
                 gray_scale_probability, do_imagenet_normalization,
                 color_jitter, label_generator, interpolation_mode, stage2_on_host_process,
                 with_raw_data):
        self.template_size = template_size
        self.search_size = search_size
        self.template_area_factor = template_area_factor
        self.search_area_factor = search_area_factor
        self.template_scale_jitter_factor = template_scale_jitter_factor
        self.search_scale_jitter_factor = search_scale_jitter_factor
        self.template_translation_jitter_factor = template_translation_jitter_factor
        self.search_translation_jitter_factor = search_translation_jitter_factor
        self.gray_scale_probability = gray_scale_probability
        self.interpolation_mode = interpolation_mode
        self.do_imagenet_normalization = do_imagenet_normalization
        if stage2_on_host_process:
            self.transform = None
        else:
            self.transform = build_TransT_image_augmentation_transformer(color_jitter, do_imagenet_normalization)
        self.gray_scale_transformer = Grayscale(3)
        self.label_generator = label_generator
        self.with_raw_data = with_raw_data

    def __call__(self, z_image, z_bbox, x_image, x_bbox, is_positive):
        miscellany = {}
        collate_miscellany = True
        if self.with_raw_data:
            miscellany['z'] = z_image
            miscellany['x'] = z_bbox
            miscellany['z_bbox'] = z_bbox
            miscellany['x_bbox'] = x_bbox
            collate_miscellany = False
        miscellany['is_positive_sample'] = is_positive
        z_image, x_image = TransT_training_image_preprocessing(z_image, x_image, self.gray_scale_transformer,
                                                               self.do_imagenet_normalization,
                                                               self.gray_scale_probability, np.random)
        z_image, z_bbox, z_context = \
            TransT_training_data_preprocessing_pipeline(z_image, z_bbox, self.template_area_factor,
                                                        self.template_size,
                                                        self.template_scale_jitter_factor,
                                                        self.template_translation_jitter_factor,
                                                        self.interpolation_mode,
                                                        self.transform)
        x_image, x_bbox, x_context = \
            TransT_training_data_preprocessing_pipeline(x_image, x_bbox, self.search_area_factor,
                                                        self.search_size,
                                                        self.search_scale_jitter_factor,
                                                        self.search_translation_jitter_factor,
                                                        self.interpolation_mode,
                                                        self.transform)

        labels = self.label_generator(x_bbox.tolist(), is_positive)

        z_image = z_image.float()
        x_image = x_image.float()

        if isinstance(labels, (list, tuple)):
            return z_image, x_image, z_context, x_context, miscellany, collate_miscellany, *labels
        else:
            return z_image, x_image, z_context, x_context, miscellany, collate_miscellany, labels
