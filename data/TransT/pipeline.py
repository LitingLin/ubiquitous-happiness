import torch
from data.operator.bbox.spatial.xyxy2xywh import bbox_xyxy2xywh
from data.operator.bbox.spatial.center import bbox_get_center_point
from data.operator.bbox.spatial.utility.aligned.image import get_image_center_point
from data.operator.image_and_bbox.align_corner.scale_and_translate import tf_scale_and_translate_numpy
from data.operator.image.mean import tf_get_image_mean
from data.operator.bbox.spatial.scale_and_translate import bbox_scale_and_translate
import math
import numpy as np
from data.operator.image.imagenet_normalize import image_torch_tensor_imagenet_normalize
from data.operator.image.rgb_to_gray import tf_image_rgb_to_gray_keep_channels
from torchvision.transforms.functional import adjust_brightness
from data.operator.image.numpy_pytorch_interop import image_numpy_to_torch_HWC_to_CHW
from data.operator.image.batchify import unbatchify
from data.operator.image.normalize import torch_image_normalize
from data.operator.bbox.spatial.xywh2cxcywh import bbox_xywh2cxcywh
from data.operator.bbox.spatial.normalize import bbox_normalize


def get_jittered_scaling_and_translate_factor(bbox, scaling, scaling_jitter_factor, translation_jitter_factor):
    scaling = scaling * torch.exp(torch.randn(2) * scaling_jitter_factor)
    bbox = bbox_xyxy2xywh(bbox)
    max_translate = (torch.tensor(bbox[2:4]) * scaling).sum() * 0.5 * translation_jitter_factor
    translate = (torch.rand(2) - 0.5) * max_translate
    return scaling, translate


def get_scaling_factor_from_area_factor(bbox, area_factor, output_size):
    bbox = bbox_xyxy2xywh(bbox)
    crop_area = area_factor * area_factor * bbox[2] * bbox[3]
    scaling = math.sqrt(output_size[0] * output_size[1] / crop_area)
    return torch.tensor((scaling, scaling))


def jittered_center_crop(image, bbox, area_factor, output_size, scaling_jitter_factor, translation_jitter_factor):
    scaling = get_scaling_factor_from_area_factor(bbox, area_factor, output_size)
    scaling, translate = get_jittered_scaling_and_translate_factor(bbox, scaling, scaling_jitter_factor, translation_jitter_factor)

    source_center = bbox_get_center_point(bbox)
    target_center = get_image_center_point(output_size)
    target_center = (torch.tensor(target_center) + translate).tolist()
    scaling = scaling.tolist()

    output_image = tf_scale_and_translate_numpy(image, output_size, scaling, source_center, target_center, tf_get_image_mean(image).numpy())
    output_bbox = bbox_scale_and_translate(bbox, scaling, source_center, target_center)
    return output_image, output_bbox


def transt_preprocessing_pipeline(image, bbox, area_factor, output_size, scaling_jitter_factor, translation_jitter_factor, brightness_jitter, gray_scale=False):
    if gray_scale:
        image = tf_image_rgb_to_gray_keep_channels(image)

    image, bbox = jittered_center_crop(image, bbox, area_factor, output_size, scaling_jitter_factor, translation_jitter_factor)

    image = unbatchify(image)
    image = image_numpy_to_torch_HWC_to_CHW(image)

    brightness_factor = np.random.uniform(max(0, 1 - brightness_jitter), 1 + brightness_jitter)
    image = adjust_brightness(image, brightness_factor)
    image = torch_image_normalize(image)

    image = image_torch_tensor_imagenet_normalize(image)

    return image, bbox
