from data.operator.image.numpy_pytorch_interop import image_numpy_to_torch
from data.operator.image.imagenet_normalize import image_torch_tensor_imagenet_normalize
from data.operator.bbox.xywh2cxcywh_normalize import bbox_xywh2cxcywh_normalize_
from data.operator.bbox.numpy_pytorch_interop import bbox_numpy_to_torch


class SimpleDataPipelineOrganizer:
    @staticmethod
    def _common_image_post_process(img):
        img = image_numpy_to_torch(img)
        img = image_torch_tensor_imagenet_normalize(img)
        return img

    @staticmethod
    def _common_post_process(img, bbox):
        h, w = img.shape[0:2]
        img = SimpleDataPipelineOrganizer._common_image_post_process(img)
        bbox_xywh2cxcywh_normalize_(bbox, (w, h))
        bbox = bbox_numpy_to_torch(bbox)
        return img, bbox

    def __call__(self, z_output, x_output):
        z, z_mask = z_output
        z = SimpleDataPipelineOrganizer._common_image_post_process(z)
        x, x_bbox = x_output
        x, x_bbox = SimpleDataPipelineOrganizer._common_post_process(x, x_bbox)
        return z, z_mask, x, x_bbox
