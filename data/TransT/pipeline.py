import torch
from data.operator.bbox.spatial.xyxy2xywh import bbox_xyxy2xywh
from data.operator.bbox.spatial.center import bbox_get_center_point
from data.operator.bbox.spatial.utility.aligned.image import get_image_center_point
from data.operator.image_and_bbox.align_corner.scale_and_translate import tf_scale_and_translate_numpy
from data.operator.image.mean import tf_get_image_mean
from data.operator.bbox.spatial.scale_and_translate import bbox_scale_and_translate
import math
from data.operator.image.rgb_to_gray import tf_image_rgb_to_gray_keep_channels
from data.operator.image.batchify import unbatchify, torch_batchify
from data.operator.bbox.spatial.utility.aligned.image import bounding_box_fit_in_image_boundary, \
    bounding_box_is_intersect_with_image
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import transforms


def get_jittered_scaling_and_translate_factor(bbox, scaling, scaling_jitter_factor, translation_jitter_factor):
    scaling = scaling / torch.exp(torch.randn(2) * scaling_jitter_factor)
    bbox = bbox_xyxy2xywh(bbox)
    max_translate = (torch.tensor(bbox[2:4]) * scaling).sum() * 0.5 * translation_jitter_factor
    translate = (torch.rand(2) - 0.5) * max_translate
    return scaling, translate


def get_scaling_factor_from_area_factor(bbox, area_factor, output_size):
    bbox = bbox_xyxy2xywh(bbox)
    w, h = bbox[2: 4]
    w_z = w + (area_factor - 1) * ((w + h) * 0.5)
    h_z = h + (area_factor - 1) * ((w + h) * 0.5)
    scaling = math.sqrt((output_size[0] * output_size[1]) / (w_z * h_z))
    return torch.tensor((scaling, scaling))


def get_scaling_and_translation_parameters(bbox, area_factor, output_size):
    scaling = get_scaling_factor_from_area_factor(bbox, area_factor, output_size)

    source_center = bbox_get_center_point(bbox)
    target_center = get_image_center_point(output_size)
    scaling = scaling.tolist()
    return scaling, source_center, target_center


def jittered_center_crop(image, bbox, area_factor, output_size, scaling_jitter_factor, translation_jitter_factor):
    while True:
        scaling = get_scaling_factor_from_area_factor(bbox, area_factor, output_size)
        scaling, translate = get_jittered_scaling_and_translate_factor(bbox, scaling, scaling_jitter_factor,
                                                                       translation_jitter_factor)

        source_center = bbox_get_center_point(bbox)
        target_center = get_image_center_point(output_size)
        target_center = (torch.tensor(target_center) - translate).tolist()
        scaling = scaling.tolist()

        output_bbox = bbox_scale_and_translate(bbox, scaling, source_center, target_center)

        if bounding_box_is_intersect_with_image(output_bbox, output_size):
            break

    output_image = tf_scale_and_translate_numpy(image, output_size, scaling, source_center, target_center,
                                                tf_get_image_mean(image).numpy())
    return output_image, output_bbox


def build_transform(color_jitter=0.4):
    # color jitter is enabled when not using AA
    if isinstance(color_jitter, (list, tuple)):
        # color jitter should be a 3-tuple/list if spec brightness/contrast/saturation
        # or 4 if also augmenting hue
        assert len(color_jitter) in (3, 4)
    else:
        # if it's a scalar, duplicate for brightness, contrast, and saturation, no hue
        color_jitter = (float(color_jitter),) * 3
    transform_list = [
        transforms.ToTensor(),
        transforms.ColorJitter(*color_jitter),
        transforms.Normalize(
            mean=torch.tensor(IMAGENET_DEFAULT_MEAN),
            std=torch.tensor(IMAGENET_DEFAULT_STD))
    ]
    return transforms.Compose(transform_list)


def build_evaluation_transform():
    transform_list = [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=torch.tensor(IMAGENET_DEFAULT_MEAN),
            std=torch.tensor(IMAGENET_DEFAULT_STD))
    ]
    return transforms.Compose(transform_list)


def transt_preprocessing_pipeline(image, bbox, output_size, curation_scaling, curation_source_center_point, curation_target_center_point, image_mean=None, transform=None):
    output_bbox = bbox_scale_and_translate(bbox, curation_scaling, curation_source_center_point, curation_target_center_point)

    if image_mean is None:
        image_mean = tf_get_image_mean(image).numpy()
    output_image = tf_scale_and_translate_numpy(image, output_size, curation_scaling, curation_source_center_point, curation_target_center_point,
                                                image_mean)

    output_image = unbatchify(output_image)

    if transform is not None:
        output_image /= 255
        output_image = transform(output_image)

    output_image = torch_batchify(output_image)

    return output_image, output_bbox, image_mean


def transt_training_preprocessing_pipeline(image, bbox, area_factor, output_size, scaling_jitter_factor,
                                           translation_jitter_factor, gray_scale=False, transform=None):
    if gray_scale:
        image = tf_image_rgb_to_gray_keep_channels(image)

    image, bbox = jittered_center_crop(image, bbox, area_factor, output_size, scaling_jitter_factor,
                                       translation_jitter_factor)
    bbox = bounding_box_fit_in_image_boundary(bbox, output_size)

    image = unbatchify(image)

    if transform is not None:
        image /= 255
        image = transform(image)

    return image, bbox
