import numpy
import numpy.random
from data.siamfc.curation_with_aug import curate_image_like_siamfc_with_aug, get_siamfc_curation_center_and_scale
from data.operator.image.numpy_pytorch_interop import image_numpy_to_torch


class SiamFCDataProcessor:
    def __init__(self, exemplar_sz, instance_sz, context=0.5, max_translation=None, max_stretch_ratio=None,
                 z_rgb_variance=None, x_rgb_variance=None, random_gray_ratio=0., label_generator=None, to_torch=False):
        self.exemplar_sz = exemplar_sz
        self.instance_sz = instance_sz
        self.context = context
        self.max_translation = max_translation
        self.max_stretch_ratio = max_stretch_ratio
        self.z_rgb_variance = None
        self.x_rgb_variance = None
        if z_rgb_variance is not None:
            self.z_rgb_variance = numpy.array(z_rgb_variance, dtype=numpy.float32)
        if x_rgb_variance is not None:
            self.x_rgb_variance = numpy.array(x_rgb_variance, dtype=numpy.float32)
        self.random_gray_ratio = random_gray_ratio
        self.label_generator = label_generator
        self.to_torch = to_torch

    def __call__(self, image_z, z_bounding_box, image_x, x_bounding_box, is_positive):
        if self.random_gray_ratio > 0.:
            do_gray_scale_augmentation = numpy.random.random_sample() < self.random_gray_ratio
        else:
            do_gray_scale_augmentation = False
        z = curate_image_like_siamfc_with_aug(image_z,
                                              *get_siamfc_curation_center_and_scale(z_bounding_box, self.context,
                                                                                    self.exemplar_sz), self.exemplar_sz,
                                              self.max_translation, self.max_stretch_ratio, self.z_rgb_variance,
                                              do_gray_scale_augmentation)
        x = curate_image_like_siamfc_with_aug(image_x,
                                              *get_siamfc_curation_center_and_scale(x_bounding_box, self.context,
                                                                                    self.exemplar_sz), self.instance_sz,
                                              self.max_translation, self.max_stretch_ratio, self.x_rgb_variance,
                                              do_gray_scale_augmentation)

        if self.to_torch:
            z = image_numpy_to_torch(z)
            x = image_numpy_to_torch(x)

        if self.label_generator:
            return (z, x), self.label_generator(is_positive)
        else:
            return z, x
