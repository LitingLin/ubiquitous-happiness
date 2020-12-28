import numpy
import numpy.random
from data.siamfc.curation import curate_image_like_siamfc_with_aug, get_siamfc_curation_center_and_scale
from data.operator.image.resize import ImageResizer
from data.operator.bbox.xywh2cxcywh_normalize import bbox_xywh2cxcywh_normalize_, bbox_denormalize_cxcywh2xywh
from data.operator.image.numpy_pytorch_interop import image_numpy_to_torch
from data.operator.bbox.numpy_pytorch_interop import bbox_numpy_to_torch, bbox_torch_to_numpy


class SiamFCZCurateXResizeProcessor:
    def __init__(self, exemplar_sz, instance_sz, context=0.5, max_translation=None, max_stretch_ratio=None,
                 z_rgb_variance=None, label_generator=None, do_image_normalize=True, return_torch_tensor=True):
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
        self.do_image_normalize = do_image_normalize
        self.return_torch_tensor = return_torch_tensor

    def get_z(self, image: numpy.ndarray, bbox: numpy.ndarray):
        z = curate_image_like_siamfc_with_aug(image,
                                              *get_siamfc_curation_center_and_scale(bbox, self.context,
                                                                                    self.exemplar_sz),
                                              self.exemplar_sz, self.max_translation, self.max_stretch_ratio,
                                              self.z_rgb_variance, False)
        if self.return_torch_tensor:
            z = image_numpy_to_torch(z)
        if self.do_image_normalize:
            z = z / 255.0
        return z

    def get_x(self, image: numpy.ndarray, bbox: numpy.ndarray):
        h, w = image.shape[0:2]
        return self.get_x_image(image), self.get_x_bbox(bbox, (w, h))

    def get_x_image(self, image: numpy.ndarray):
        image = self.x_resizer.do_image(image)
        if self.return_torch_tensor:
            image = image_numpy_to_torch(image)
        if self.do_image_normalize:
            image = image / 255.0
        return image

    def get_x_bbox(self, bbox: numpy.ndarray, image_size):
        bbox = self.x_resizer.do_bbox(bbox, image_size)
        bbox = bbox_xywh2cxcywh_normalize_(bbox, (self.instance_sz, self.instance_sz))
        if self.return_torch_tensor:
            bbox = bbox_numpy_to_torch(bbox)
        return bbox

    def reverse_x_bbox(self, bbox, origin_image_size):
        if self.return_torch_tensor:
            bbox = bbox_torch_to_numpy(bbox)
        bbox = bbox_denormalize_cxcywh2xywh(bbox, (self.instance_sz, self.instance_sz))
        return self.x_resizer.reverse_do_bbox_(bbox, origin_image_size)

    def __call__(self, image_z: numpy.ndarray, z_bounding_box: numpy.ndarray, image_x: numpy.ndarray, x_bounding_box: numpy.ndarray, is_positive):
        return (self.get_z(image_z, z_bounding_box), *self.get_x(image_x, x_bounding_box))
