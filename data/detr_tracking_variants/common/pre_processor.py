from data.operator.bbox.align_corner.xywh2cxcywh_normalize import bbox_denormalize_cxcywh2xywh
from data.operator.bbox.numpy_pytorch_interop import bbox_torch_to_numpy


class PreProcessor_BoundingBoxToNumpyToXYWH:
    def __call__(self, bbox, origin_image_size):
        bbox = bbox_torch_to_numpy(bbox)
        return bbox_denormalize_cxcywh2xywh(bbox, origin_image_size)
