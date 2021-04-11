import numpy
from data.siamfc.curation import siamfc_z_curation_with_bbox
from data.operator.image.module.resize import ImageResizer
from data.operator.bbox.aligned.xywh2cxcywh_normalize import bbox_xywh2cxcywh_normalize_, bbox_denormalize_cxcywh2xywh
from data.operator.image.numpy_pytorch_interop import image_numpy_to_torch
from data.operator.bbox.numpy_pytorch_interop import bbox_numpy_to_torch, bbox_torch_to_numpy


class SiamFC_Z_Curate_BBOX_XYWH_X_Resize_BBOX_CXCYWHNormalized_Processor:
    def __init__(self, exemplar_sz, instance_sz, context=0.5, do_image_normalize=True, return_torch_tensor=True):
        self.exemplar_sz = exemplar_sz
        self.instance_sz = instance_sz
        self.context = context
        self.x_resizer = ImageResizer(instance_sz)
        self.do_image_normalize = do_image_normalize
        self.return_torch_tensor = return_torch_tensor

    def get_z(self, image: numpy.ndarray, bbox: numpy.ndarray):
        z, z_bbox = siamfc_z_curation_with_bbox(image, bbox, self.context, self.exemplar_sz)
        if self.return_torch_tensor:
            z = image_numpy_to_torch(z)
            z_bbox = bbox_numpy_to_torch(z_bbox)
        if self.do_image_normalize:
            z = z / 255.0
        return z, z_bbox

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
        return (*self.get_z(image_z, z_bounding_box), *self.get_x(image_x, x_bounding_box))
