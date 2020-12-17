import numpy
import numpy.random
from data.siamfc.curation import curate_image_like_siamfc_with_aug, get_siamfc_curation_center_and_scale
from data.augmentation.resize import ImageResizer


class SiamFCZCurateXResizeProcessor:
    def __init__(self, exemplar_sz, instance_sz, context=0.5, max_translation=None, max_stretch_ratio=None,
                 z_rgb_variance=None, label_generator=None):
        self.exemplar_sz = exemplar_sz
        self.instance_sz = instance_sz
        self.context = context
        self.max_translation = max_translation
        self.max_stretch_ratio = max_stretch_ratio
        self.z_rgb_variance = None
        self.x_rgb_variance = None
        if z_rgb_variance is not None:
            self.z_rgb_variance = numpy.array(z_rgb_variance, dtype=numpy.float32)
        self.label_generator = label_generator
        self.x_resizer = ImageResizer(instance_sz)

    def __call__(self, image_z, z_bounding_box, image_x, x_bounding_box, is_positive):
        z = curate_image_like_siamfc_with_aug(image_z,
                                              *get_siamfc_curation_center_and_scale(z_bounding_box, self.context,
                                                                                    self.exemplar_sz), self.exemplar_sz,
                                              self.max_translation, self.max_stretch_ratio, self.z_rgb_variance,
                                              False)

        image_x, x_bounding_box = self.x_resizer(image_x, x_bounding_box)

        return z, image_x, x_bounding_box
