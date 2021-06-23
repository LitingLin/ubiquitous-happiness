import torch
from data.operator.bbox.spatial.xyxy2xywh import bbox_xyxy2xywh
from data.operator.bbox.spatial.center import bbox_get_center_point
from data.operator.bbox.spatial.utility.aligned.image import get_image_center_point
from data.operator.bbox.spatial.scale_and_translate import bbox_scale_and_translate
from data.operator.image_and_bbox.align_corner.vectorized.pytorch.scale_and_translate import torch_scale_and_translate_align_corners
from data.operator.image.pytorch.mean import get_image_mean_chw, get_image_mean
import math
from data.operator.bbox.spatial.vectorized.torch.utility.aligned.image import bbox_restrict_in_image_boundary_
from data.operator.bbox.spatial.utility.aligned.image import bounding_box_is_intersect_with_image
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
    source_center = torch.tensor(source_center)
    target_center = torch.tensor(target_center)
    return scaling, source_center, target_center


def jittered_center_crop(image, bbox, area_factor, output_size, scaling_jitter_factor, translation_jitter_factor):
    while True:
        scaling = get_scaling_factor_from_area_factor(bbox, area_factor, output_size)
        scaling, translate = get_jittered_scaling_and_translate_factor(bbox, scaling, scaling_jitter_factor,
                                                                       translation_jitter_factor)

        source_center = bbox_get_center_point(bbox)
        target_center = get_image_center_point(output_size)
        target_center = (torch.tensor(target_center) - translate)

        output_bbox = bbox_scale_and_translate(bbox, scaling, source_center, target_center)

        if bounding_box_is_intersect_with_image(output_bbox, output_size):
            break
    source_center = torch.tensor(source_center)
    output_bbox = torch.tensor(output_bbox)
    output_image, _ = torch_scale_and_translate_align_corners(image, output_size, scaling, source_center, target_center, get_image_mean_chw(image))
    return output_image, output_bbox


def build_TransT_image_augmentation_transformer(color_jitter=0.4):
    # color jitter is enabled when not using AA
    if isinstance(color_jitter, (list, tuple)):
        # color jitter should be a 3-tuple/list if spec brightness/contrast/saturation
        # or 4 if also augmenting hue
        assert len(color_jitter) in (3, 4)
    else:
        # if it's a scalar, duplicate for brightness, contrast, and saturation, no hue
        color_jitter = (float(color_jitter),) * 3
    transform_list = [
        transforms.ColorJitter(*color_jitter),
        transforms.Normalize(
            mean=torch.tensor(IMAGENET_DEFAULT_MEAN),
            std=torch.tensor(IMAGENET_DEFAULT_STD))
    ]
    return transforms.Compose(transform_list)


def build_evaluation_transform():
    return transforms.Normalize(
            mean=torch.tensor(IMAGENET_DEFAULT_MEAN),
            std=torch.tensor(IMAGENET_DEFAULT_STD))


def transt_data_processing_evaluation_pipeline(image, bbox, output_size, curation_scaling, curation_source_center_point, curation_target_center_point, image_mean=None, transform=None):
    image = image.float() / 255.
    output_bbox = bbox_scale_and_translate(bbox, curation_scaling, curation_source_center_point, curation_target_center_point)
    output_bbox = torch.tensor(output_bbox)

    if image_mean is None:
        image_mean = get_image_mean(image)
    curation_scaling = curation_scaling.to(image.device, non_blocking=True)
    curation_source_center_point = curation_source_center_point.to(image.device, non_blocking=True)
    curation_target_center_point = curation_target_center_point.to(image.device, non_blocking=True)
    output_image, _ = torch_scale_and_translate_align_corners(image, output_size, curation_scaling, curation_source_center_point, curation_target_center_point, image_mean)

    if transform is not None:
        output_image = transform(output_image)

    return output_image, output_bbox, image_mean


def TransT_training_data_preprocessing_pipeline(image, bbox, area_factor, output_size, scaling_jitter_factor,
                                                translation_jitter_factor, transform=None):
    image, bbox = jittered_center_crop(image, bbox, area_factor, output_size, scaling_jitter_factor,
                                       translation_jitter_factor)
    bbox_restrict_in_image_boundary_(bbox, output_size)

    if transform is not None:
        image = transform(image)

    return image, bbox


def _transt_data_pre_processing_train_pipeline(image, gray_scale_transformer):
    if gray_scale_transformer is not None:
        image = gray_scale_transformer(image)
    return image.float() / 255.


def TransT_training_image_preprocessing(z_image, x_image, gray_scale_transformer, gray_scale_probability, rng_engine):
    if rng_engine.random() > gray_scale_probability:
        gray_scale_transformer = None
    if id(x_image) != id(z_image):
        z_image = _transt_data_pre_processing_train_pipeline(z_image, gray_scale_transformer)
        x_image = _transt_data_pre_processing_train_pipeline(x_image, gray_scale_transformer)
    else:
        x_image = z_image = _transt_data_pre_processing_train_pipeline(z_image, gray_scale_transformer)
    return z_image, x_image
