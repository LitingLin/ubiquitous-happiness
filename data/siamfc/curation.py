import numpy as np
import cv2
from native_extension import InterpolationMethod, RGBImageTranslateAndScaleWithBoundingBox, RGBImageTranslateAndScale


def curate_image_like_siamfc(image, center, scale, out_size):
    avg_color = np.round(cv2.mean(image)[0:3]).astype(np.uint8)
    out_center = [out_size / 2, out_size / 2]
    interp_method = InterpolationMethod.INTER_LINEAR
    image, image_bbox = RGBImageTranslateAndScale(image, [out_size, out_size], center, out_center, scale, avg_color, interp_method)
    return image, image_bbox


def curate_image_like_siamfc_with_bbox(image, center, scale, out_size):
    avg_color = np.round(cv2.mean(image)[0:3]).astype(np.uint8)
    out_center = [out_size / 2, out_size / 2]
    interp_method = InterpolationMethod.INTER_LINEAR
    image, image_bbox = RGBImageTranslateAndScaleWithBoundingBox(image, [out_size, out_size], center, out_center, scale, avg_color, interp_method)
    return image, image_bbox


def get_siamfc_curation_center_and_scale(bounding_box, context, exemplar_size):
    assert len(bounding_box) == 4
    bounding_box = np.array(bounding_box, dtype=np.float32)
    target_size = bounding_box[2:4]

    context = context * np.sum(target_size)
    scale = np.sqrt(exemplar_size ** 2 / np.prod(target_size + context))

    scale = [scale, scale]

    center = [bounding_box[0] - 1 + bounding_box[2] / 2, bounding_box[1] - 1 + bounding_box[3] / 2]
    return center, scale


def siamfc_z_curation_with_bbox(image, bounding_box, context, exemplar_size):
    return curate_image_like_siamfc_with_bbox(image, *get_siamfc_curation_center_and_scale(bounding_box, context,
                                                                                          exemplar_size), exemplar_size)


def siamfc_x_curation_with_bbox(image, bounding_box, context, exemplar_size, instance_size):
    return curate_image_like_siamfc_with_bbox(image, *get_siamfc_curation_center_and_scale(bounding_box, context,
                                                                                          exemplar_size), instance_size)
