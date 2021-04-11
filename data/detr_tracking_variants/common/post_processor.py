from data.operator.image.numpy_pytorch_interop import image_numpy_to_torch
from data.operator.image.imagenet_normalize import image_torch_tensor_imagenet_normalize
from data.operator.bbox.aligned.xywh2cxcywh_normalize import bbox_xywh2cxcywh_normalize_
from data.operator.bbox.numpy_pytorch_interop import bbox_numpy_to_torch
import torch


def _common_image_post_process(img):
    img = image_numpy_to_torch(img)
    img = image_torch_tensor_imagenet_normalize(img)
    return img


def _common_post_process(img, bbox):
    h, w = img.shape[0:2]
    img = _common_image_post_process(img)
    bbox_xywh2cxcywh_normalize_(bbox, (w, h))
    bbox = bbox_numpy_to_torch(bbox)
    return img, bbox


class PostProcessor_ImageToTorchImagenetNormalizationAnnotationToTorch:
    def __call__(self, input_):
        img, anno = input_
        img = _common_image_post_process(img)
        return img, torch.tensor(anno)


class PostProcessor_ImageToTorchImagenetNormalizationNoAnnotation:
    def __call__(self, img):
        img = _common_image_post_process(img)
        return img


class PostProcessor_ImageToTorchImagenetNormalizationBoundingBoxToCXCYWHNormalizedToTorch:
    def __call__(self, input_):
        img, bbox = input_
        img, bbox = _common_post_process(img, bbox)
        return img, bbox


class PostProcessor_ImageToTorchImagenetNormalizationMaskGenerating:
    def __call__(self, input_):
        img, _ = input_
        h, w = img.shape[0: 2]
        img = _common_image_post_process(img)
        mask = torch.zeros((h, w), dtype=torch.bool)
        return img, mask
